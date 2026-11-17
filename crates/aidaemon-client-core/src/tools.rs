//! A turn's tool calls as the owner reads them: what each call did and
//! what it worked on ("Ran `cargo test -q`", "Read notes.md"), and one
//! line for the whole turn ("Ran 3 commands, changed 2 files · 12s").
//!
//! The kind of work comes from the daemon, typed ([`ToolKind`]), never
//! from the tool's name; what a call worked on is the argument that says
//! it (`command`, `path`, `query`, …).

use aidaemon_client_protocol::{ToolCallInfo, ToolKind};

/// One call as a row reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    /// What it did, in a word or two: "Ran", "Read", "Looked up"; for a
    /// tool without a kind of its own, its name in words.
    pub verb: String,
    /// What it worked on, on one line: a command, a path, a query.
    pub subject: Option<String>,
    /// Whether its input says more than the row does (shown when opened).
    pub more_input: bool,
    /// Whether its output is a command's (shown as a terminal).
    pub command_output: bool,
}

/// How long a subject may be before it ends in "…".
const SUBJECT_CHARS: usize = 120;

/// The arguments that name what a call of each kind worked on, best first.
fn subject_keys(kind: Option<ToolKind>) -> &'static [&'static str] {
    match kind {
        Some(ToolKind::Run) => &["command", "cmd", "script"],
        Some(ToolKind::Read | ToolKind::Write) => &["path", "file_path", "file", "paths"],
        Some(ToolKind::Search) => &["pattern", "query", "glob", "path"],
        Some(ToolKind::Web) => &["query", "url"],
        Some(ToolKind::Delegate) => &["mission", "task", "goal", "description"],
        Some(ToolKind::Remind) => &["when", "task", "name"],
        _ => &[
            "command", "path", "query", "url", "pattern", "mission", "task", "name", "action",
        ],
    }
}

/// The past-tense verb for a kind; `None` for a kind that says nothing of
/// its own ([`ToolKind::Check`], [`ToolKind::Change`]).
fn verb(kind: ToolKind) -> Option<&'static str> {
    Some(match kind {
        ToolKind::Read => "Read",
        ToolKind::Search => "Searched",
        ToolKind::Write => "Changed",
        ToolKind::Run => "Ran",
        ToolKind::Web => "Looked up",
        ToolKind::Delegate => "Asked a helper",
        ToolKind::Recall => "Checked memory",
        ToolKind::Remember => "Updated memory",
        ToolKind::Remind => "Set a reminder",
        ToolKind::Check | ToolKind::Change | ToolKind::Unknown => return None,
    })
}

/// A tool's name in words: `track_requirements` → "Track requirements".
fn name_in_words(name: &str) -> String {
    let words = name.replace(['_', '-'], " ");
    let mut chars = words.trim().chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => "Tool".into(),
    }
}

/// `text` on one line, trimmed to [`SUBJECT_CHARS`].
fn one_line(text: &str) -> Option<String> {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.is_empty() {
        return None;
    }
    if line.chars().count() <= SUBJECT_CHARS {
        return Some(line);
    }
    let cut: String = line.chars().take(SUBJECT_CHARS - 1).collect();
    Some(format!("{}…", cut.trim_end()))
}

/// The string value of `key` in a JSON object's text, read even when the
/// text was trimmed and no longer parses: `"key": "value"`.
fn scan_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let after = &json[json.find(&pattern)? + pattern.len()..];
    let after = after.trim_start().strip_prefix(':')?.trim_start();
    let body = after.strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    // Trimmed inside the value: what there is of it.
    Some(out)
}

/// What a call worked on and whether its input says more: the first of
/// its kind's arguments with a value. Input that is not a JSON object (a
/// live status summary) is the subject itself.
fn subject(kind: Option<ToolKind>, input: &str) -> (Option<String>, bool) {
    let input = input.trim();
    if input.is_empty() {
        return (None, false);
    }
    if !input.starts_with('{') {
        // A live summary marks a command as code: `cargo test`.
        let bare = input
            .strip_prefix('`')
            .and_then(|s| s.strip_suffix('`'))
            .unwrap_or(input);
        return (one_line(bare), false);
    }
    let parsed = serde_json::from_str::<serde_json::Value>(input).ok();
    let object = parsed.as_ref().and_then(|v| v.as_object());
    for key in subject_keys(kind) {
        let value = match object {
            Some(object) => object.get(*key).and_then(|v| match v {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Array(items) => {
                    let names: Vec<&str> = items.iter().filter_map(|i| i.as_str()).collect();
                    (!names.is_empty()).then(|| names.join(", "))
                }
                _ => None,
            }),
            None => scan_string(input, key),
        };
        if let Some(subject) = value.as_deref().and_then(one_line) {
            // More to see: any argument besides this one and the action.
            let more = match object {
                Some(object) => object.keys().any(|k| k != key && k != "action"),
                None => true,
            };
            return (Some(subject), more);
        }
    }
    (None, true)
}

/// One call as a row reads it.
pub fn step(call: &ToolCallInfo) -> Step {
    let kind = call.kind;
    let (subject, more_input) = subject(kind, &call.input);
    let verb = kind
        .and_then(verb)
        .map(str::to_owned)
        .unwrap_or_else(|| name_in_words(&call.name));
    Step {
        verb,
        subject,
        more_input,
        command_output: kind == Some(ToolKind::Run),
    }
}

/// `n` of a noun: "1 file", "3 files".
fn count(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// One phrase per kind of work, in the order each first came: "ran 3
/// commands", "changed 2 files". Files count once however often touched.
fn phrases(tools: &[String], calls: &[ToolCallInfo]) -> Vec<String> {
    // (kind, calls, distinct subjects)
    let mut groups: Vec<(Option<ToolKind>, usize, Vec<String>)> = Vec::new();
    for (n, name) in tools.iter().enumerate() {
        let call = calls.get(n).filter(|c| &c.name == name);
        let kind = call.and_then(|c| c.kind).filter(|k| verb(*k).is_some());
        let subject = call.and_then(|c| subject(kind, &c.input).0);
        let group = match groups.iter_mut().position(|(k, ..)| *k == kind) {
            Some(i) => &mut groups[i],
            None => {
                groups.push((kind, 0, Vec::new()));
                groups.last_mut().expect("just pushed")
            }
        };
        group.1 += 1;
        if let Some(subject) = subject
            && !group.2.contains(&subject)
        {
            group.2.push(subject);
        }
    }
    groups
        .into_iter()
        .map(|(kind, calls, subjects)| {
            // Files count once each when their paths are known.
            let files = if subjects.is_empty() {
                calls
            } else {
                subjects.len()
            };
            match kind {
                Some(ToolKind::Run) => format!("ran {}", count(calls, "command", "commands")),
                Some(ToolKind::Read) => format!("read {}", count(files, "file", "files")),
                Some(ToolKind::Write) => format!("changed {}", count(files, "file", "files")),
                Some(ToolKind::Search) => match calls {
                    1 => "searched files".into(),
                    n => format!("searched files {n} times"),
                },
                Some(ToolKind::Web) => match calls {
                    1 => "looked something up".into(),
                    n => format!("looked up {n} things"),
                },
                Some(ToolKind::Delegate) => format!("asked {}", count(calls, "helper", "helpers")),
                Some(ToolKind::Recall) => "checked memory".into(),
                Some(ToolKind::Remember) => "updated memory".into(),
                Some(ToolKind::Remind) => format!("set {}", count(calls, "reminder", "reminders")),
                _ => count(calls, "other step", "other steps"),
            }
        })
        .collect()
}

/// "a, b and c".
fn and_list(parts: &[String]) -> String {
    match parts {
        [] => String::new(),
        [one] => one.clone(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    }
}

/// The whole turn on one line: "Ran 3 commands, changed 2 files · 12s";
/// while it runs, "Ran 1 command so far". Only other steps (no kind from
/// an older daemon): "Used 4 tools".
pub fn summary(tools: &[String], calls: &[ToolCallInfo], secs: u64, live: bool) -> String {
    let phrases = phrases(tools, calls);
    let only_other = phrases.len() == 1 && phrases[0].ends_with("other step")
        || phrases.len() == 1 && phrases[0].ends_with("other steps");
    let mut line = if only_other || phrases.is_empty() {
        format!("Used {}", count(tools.len(), "tool", "tools"))
    } else {
        // At most three kinds; the rest are counted as other steps.
        let shown: Vec<String> = if phrases.len() > 3 {
            let rest = phrases.len() - 2;
            let mut first: Vec<String> = phrases[..2].to_vec();
            first.push(count(rest, "more kind of step", "more kinds of step"));
            first
        } else {
            phrases
        };
        let mut text = and_list(&shown);
        if let Some(first) = text.get(..1) {
            text = format!("{}{}", first.to_uppercase(), &text[1..]);
        }
        text
    };
    if live {
        line.push_str(" so far");
    } else if secs > 0 {
        line.push_str(&match secs {
            1..60 => format!(" · {secs}s"),
            _ => format!(" · {}m {}s", secs / 60, secs % 60),
        });
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, kind: Option<ToolKind>, input: &str) -> ToolCallInfo {
        ToolCallInfo {
            name: name.into(),
            input: input.into(),
            output: String::new(),
            kind,
        }
    }

    #[test]
    fn a_call_reads_as_what_it_did_to_what() {
        let run = step(&call(
            "terminal",
            Some(ToolKind::Run),
            "{\n  \"action\": \"run\",\n  \"command\": \"cargo test -q\"\n}",
        ));
        assert_eq!(run.verb, "Ran");
        assert_eq!(run.subject.as_deref(), Some("cargo test -q"));
        assert!(!run.more_input, "the action alone says nothing more");
        assert!(run.command_output);

        let edit = step(&call(
            "edit_file",
            Some(ToolKind::Write),
            r#"{"path": "src/lib.rs", "old_text": "a", "new_text": "b"}"#,
        ));
        assert_eq!(edit.verb, "Changed");
        assert_eq!(edit.subject.as_deref(), Some("src/lib.rs"));
        assert!(edit.more_input);

        // No kind of its own: the tool's name in words.
        let other = step(&call(
            "track_requirements",
            Some(ToolKind::Check),
            r#"{"action": "list"}"#,
        ));
        assert_eq!(other.verb, "Track requirements");
        assert_eq!(other.subject.as_deref(), Some("list"));
    }

    #[test]
    fn a_trimmed_input_and_a_live_summary_still_name_their_subject() {
        let trimmed = step(&call(
            "read_file",
            Some(ToolKind::Read),
            "{\n  \"path\": \"notes/a.md\",\n  \"start_line\": 1, … (trimmed",
        ));
        assert_eq!(trimmed.subject.as_deref(), Some("notes/a.md"));
        let live = step(&call(
            "running a command",
            Some(ToolKind::Run),
            "`npm test`",
        ));
        assert_eq!(live.subject.as_deref(), Some("npm test"));
        let long = step(&call(
            "terminal",
            Some(ToolKind::Run),
            &format!(r#"{{"command": "{}"}}"#, "x".repeat(300)),
        ));
        assert!(long.subject.unwrap().ends_with('…'));
    }

    #[test]
    fn the_turn_line_counts_work_by_kind_and_files_once_each() {
        let calls = vec![
            call("terminal", Some(ToolKind::Run), r#"{"command": "ls"}"#),
            call("edit_file", Some(ToolKind::Write), r#"{"path": "a.rs"}"#),
            call("edit_file", Some(ToolKind::Write), r#"{"path": "a.rs"}"#),
            call(
                "terminal",
                Some(ToolKind::Run),
                r#"{"command": "cargo test"}"#,
            ),
            call("write_file", Some(ToolKind::Write), r#"{"path": "b.rs"}"#),
            call("track_requirements", Some(ToolKind::Check), "{}"),
        ];
        let tools: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
        assert_eq!(
            summary(&tools, &calls, 42, false),
            "Ran 2 commands, changed 2 files and 1 other step · 42s"
        );
        assert_eq!(
            summary(&tools[..1], &calls[..1], 0, true),
            "Ran 1 command so far"
        );
    }

    #[test]
    fn without_kinds_the_line_counts_tools() {
        let tools = vec!["a".to_string(), "b".to_string()];
        assert_eq!(summary(&tools, &[], 75, false), "Used 2 tools · 1m 15s");
    }

    #[test]
    fn many_kinds_keep_the_line_short() {
        let calls = vec![
            call("r", Some(ToolKind::Read), ""),
            call("w", Some(ToolKind::Web), ""),
            call("s", Some(ToolKind::Search), ""),
            call("d", Some(ToolKind::Delegate), ""),
        ];
        let tools: Vec<String> = calls.iter().map(|c| c.name.clone()).collect();
        assert_eq!(
            summary(&tools, &calls, 0, false),
            "Read 1 file, looked something up and 2 more kinds of step"
        );
    }
}
