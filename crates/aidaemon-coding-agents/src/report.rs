//! What a coding agent ends its run with, in fields: the one shape every
//! agent is asked for ([`schema`]) where its protocol can be held to a
//! schema (Claude Code's `--json-schema`, Codex's output schema). An agent
//! that can't be held to one reports text alone, and says so
//! ([`crate::AgentEvent::Reported`] with no report).
//!
//! A report is the agent's claim. The embedding agent checks each claim
//! against its own records (what changed on disk, which commands ran and
//! how they exited) before it trusts one.

use serde_json::{Value, json};

/// The agent's account of its run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// What it did or found, in a few sentences.
    pub summary: String,
    /// The files it says it changed, as paths.
    pub changed_files: Vec<String>,
    /// The checks it says it ran (tests, builds, linters) and how each went.
    pub checks: Vec<Check>,
    /// What it could not settle on its own.
    pub open_questions: Vec<String>,
    /// It needs the owner before the work can go on.
    pub needs_owner: bool,
}

/// One check an agent says it ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    /// The command line it ran.
    pub command: String,
    pub passed: bool,
}

/// The JSON Schema a report follows: every field required and nothing
/// else allowed, the strict form both Claude Code and Codex accept.
pub fn schema() -> Value {
    let strings = json!({"type": "array", "items": {"type": "string"}});
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["summary", "changed_files", "checks", "open_questions", "needs_owner"],
        "properties": {
            "summary": {
                "type": "string",
                "description": "What you did or found, in at most three sentences."
            },
            "changed_files": strings.clone(),
            "checks": {
                "type": "array",
                "description": "Each test, build or lint command you ran, and whether it passed. Only commands you ran: a check you made by reading a file is not one.",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["command", "passed"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command line exactly as you ran it."
                        },
                        "passed": {"type": "boolean"}
                    }
                }
            },
            "open_questions": strings,
            "needs_owner": {
                "type": "boolean",
                "description": "True only when the work can't go on without the owner."
            }
        }
    })
}

impl Report {
    /// The report in `value` when it has the schema's shape: a summary at
    /// least; a field of the wrong type makes it no report at all.
    pub fn from_value(value: &Value) -> Option<Self> {
        let object = value.as_object()?;
        let strings = |key: &str| -> Option<Vec<String>> {
            match object.get(key) {
                None | Some(Value::Null) => Some(Vec::new()),
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|item| item.as_str().map(str::to_string))
                    .collect(),
                Some(_) => None,
            }
        };
        let checks = match object.get("checks") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => items
                .iter()
                .map(|item| {
                    Some(Check {
                        command: item.get("command")?.as_str()?.to_string(),
                        passed: item.get("passed")?.as_bool()?,
                    })
                })
                .collect::<Option<Vec<_>>>()?,
            Some(_) => return None,
        };
        Some(Self {
            summary: object.get("summary")?.as_str()?.trim().to_string(),
            changed_files: strings("changed_files")?,
            checks,
            open_questions: strings("open_questions")?,
            needs_owner: match object.get("needs_owner") {
                None | Some(Value::Null) => false,
                Some(value) => value.as_bool()?,
            },
        })
        .filter(|report| !report.summary.is_empty())
    }

    /// The report in a final message's text: the whole text is its JSON
    /// (what a schema-held agent writes). `None` when it isn't.
    pub fn from_text(text: &str) -> Option<Self> {
        serde_json::from_str::<Value>(text.trim())
            .ok()
            .and_then(|value| Self::from_value(&value))
    }

    /// The report as text a person or a model reads: its summary, then the
    /// files it says it changed, the checks it says it ran, what it left
    /// open and whether it waits on the owner. Its claims, as the agent
    /// made them.
    pub fn to_text(&self) -> String {
        let mut text = self.summary.trim().to_string();
        if !self.changed_files.is_empty() {
            text.push_str("\n\nChanged files:");
            for path in &self.changed_files {
                text.push_str(&format!("\n- {}", path.trim()));
            }
        }
        if !self.checks.is_empty() {
            text.push_str("\n\nChecks:");
            for check in &self.checks {
                let how = if check.passed { "passed" } else { "failed" };
                text.push_str(&format!("\n- {} ({how})", check.command.trim()));
            }
        }
        if !self.open_questions.is_empty() {
            text.push_str("\n\nOpen questions:");
            for question in &self.open_questions {
                text.push_str(&format!("\n- {}", question.trim()));
            }
        }
        if self.needs_owner {
            text.push_str("\n\nIt needs the owner before the work can go on.");
        }
        text
    }

    /// Its stored form.
    pub fn to_value(&self) -> Value {
        json!({
            "summary": self.summary,
            "changed_files": self.changed_files,
            "checks": self.checks.iter().map(|check| json!({
                "command": check.command,
                "passed": check.passed,
            })).collect::<Vec<_>>(),
            "open_questions": self.open_questions,
            "needs_owner": self.needs_owner,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_reads_back_from_its_stored_form_and_a_bad_one_is_none() {
        let report = Report {
            summary: "Added a test".into(),
            changed_files: vec!["tests/test_slug.py".into()],
            checks: vec![Check {
                command: "pytest -q".into(),
                passed: true,
            }],
            open_questions: Vec::new(),
            needs_owner: false,
        };
        assert_eq!(Report::from_value(&report.to_value()), Some(report.clone()));
        assert_eq!(
            Report::from_text(&report.to_value().to_string()),
            Some(report)
        );
        assert_eq!(
            Report::from_text("Added a test."),
            None,
            "prose is no report"
        );
        assert_eq!(
            Report::from_value(&json!({"summary": "x", "checks": "all passed"})),
            None,
            "a field of the wrong type"
        );
        assert_eq!(Report::from_value(&json!({"summary": "  "})), None);
        assert_eq!(schema()["required"].as_array().map(Vec::len), Some(5));
    }

    #[test]
    fn a_report_reads_as_text_with_only_the_parts_it_has() {
        let report = Report {
            summary: "Renamed the button to Send.".into(),
            changed_files: vec!["form.html".into()],
            checks: vec![Check {
                command: "npm test".into(),
                passed: false,
            }],
            open_questions: vec!["Keep the old label in the French copy?".into()],
            needs_owner: true,
        };
        assert_eq!(
            report.to_text(),
            "Renamed the button to Send.\n\nChanged files:\n- form.html\n\nChecks:\n- npm test (failed)\n\nOpen questions:\n- Keep the old label in the French copy?\n\nIt needs the owner before the work can go on."
        );
        let bare = Report {
            summary: "Nothing to change.".into(),
            ..Report::default()
        };
        assert_eq!(bare.to_text(), "Nothing to change.");
    }
}
