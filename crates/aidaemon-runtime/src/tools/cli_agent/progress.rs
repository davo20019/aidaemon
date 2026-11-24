//! What a coding agent run got done before it ended, read back from the
//! events kept as it ran (`cli_agent_events`, written step by step): its
//! plan, the files it changed, and its last steps. A run that never
//! finished (a daemon restart stopped it, it failed part way) leaves this,
//! so its card can say what it changed and a follow-up can tell the agent
//! where it stopped, whatever the agent kept of its own session.
//!
//! The events' data is the daemon's own stored form
//! (`aidaemon_coding_agents::AgentEvent::to_record`), a machine format it
//! wrote itself.

use aidaemon_tools::services::CliAgentEvent;
use serde_json::Value;

/// What the daemon wrote in place of a cut-off run's output when a restart
/// stopped it (a marker the daemon itself writes).
pub const STOPPED_BY_RESTART: &str = "Stopped by a daemon restart before it finished.";

/// Most recent steps a progress note names.
const LAST_STEPS: usize = 8;
/// Most files a progress note names.
const MAX_FILES: usize = 40;

/// One plan item and where it stood.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanItem {
    pub text: String,
    /// "pending", "in_progress" or "done", as kept.
    pub status: String,
}

/// A run's progress as its kept events record it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunProgress {
    /// Its plan, as last reported.
    pub plan: Vec<PlanItem>,
    /// The files it edited, first edit first (an edit step that failed is
    /// left out).
    pub edited: Vec<String>,
    /// How many of its steps ended without failing.
    pub steps_done: usize,
    /// Its last steps, in its own words, oldest first.
    pub last_steps: Vec<String>,
}

impl RunProgress {
    /// Read `events` (oldest first).
    pub fn of(events: &[CliAgentEvent]) -> Self {
        let mut progress = Self::default();
        // Each step by the agent's own id: its words and the file it edits.
        let mut steps: std::collections::HashMap<String, (String, Option<String>)> =
            Default::default();
        for event in events {
            let Ok(data) = serde_json::from_str::<Value>(&event.data) else {
                continue;
            };
            let text = |key: &str| data.get(key).and_then(Value::as_str).map(str::to_string);
            match event.kind.as_str() {
                "step" => {
                    let edits = data
                        .get("tool")
                        .filter(|tool| tool.get("kind").and_then(Value::as_str) == Some("edit"))
                        .and_then(|tool| tool.get("path"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    if let Some(id) = text("id") {
                        steps.insert(id, (text("words").unwrap_or_default(), edits));
                    }
                }
                "step_ended" => {
                    let failed = data.get("failed").and_then(Value::as_bool) == Some(true);
                    let Some((words, edits)) = text("id").and_then(|id| steps.remove(&id)) else {
                        continue;
                    };
                    if failed {
                        continue;
                    }
                    progress.steps_done += 1;
                    if !words.trim().is_empty() {
                        progress.last_steps.push(words);
                        if progress.last_steps.len() > LAST_STEPS {
                            progress.last_steps.remove(0);
                        }
                    }
                    if let Some(path) = edits
                        && !progress.edited.contains(&path)
                    {
                        progress.edited.push(path);
                    }
                }
                "plan" => {
                    progress.plan = data
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|item| {
                            Some(PlanItem {
                                text: item.get("text")?.as_str()?.to_string(),
                                status: item
                                    .get("status")
                                    .and_then(Value::as_str)
                                    .unwrap_or("pending")
                                    .to_string(),
                            })
                        })
                        .collect();
                }
                _ => {}
            }
        }
        progress
    }

    /// Whether it records anything done.
    pub fn is_empty(&self) -> bool {
        self.plan.is_empty() && self.edited.is_empty() && self.steps_done == 0
    }

    /// The note a follow-up's agent gets about run `run`, which ended
    /// before it finished (`why`, in plain words): what it had done, so the
    /// agent checks that and goes on instead of starting over.
    pub fn note(&self, run: &str, why: &str) -> String {
        let mut note = format!(
            "[Where your last turn stopped] Your previous turn in this session (run {run}) did \
             not finish: {why} Your own session may not hold all of it. What the daemon \
             recorded of it as it ran:"
        );
        if !self.plan.is_empty() {
            note.push_str("\n- Your plan:");
            for item in &self.plan {
                let mark = match item.status.as_str() {
                    "done" => "done",
                    "in_progress" => "in progress when it stopped",
                    _ => "not started",
                };
                note.push_str(&format!("\n  - [{mark}] {}", item.text));
            }
        }
        if self.edited.is_empty() {
            note.push_str("\n- Files you changed: none recorded.");
        } else {
            note.push_str("\n- Files you changed (their changes are still there):");
            for path in self.edited.iter().take(MAX_FILES) {
                note.push_str(&format!("\n  - {path}"));
            }
            if self.edited.len() > MAX_FILES {
                note.push_str(&format!("\n  - and {} more", self.edited.len() - MAX_FILES));
            }
        }
        note.push_str(&format!("\n- Steps finished: {}.", self.steps_done));
        if !self.last_steps.is_empty() {
            note.push_str(" The last ones, oldest first:");
            for step in &self.last_steps {
                note.push_str(&format!("\n  - {step}"));
            }
        }
        note.push_str(
            "\nCheck the current state of those files before redoing anything, then go on from \
             there with the message below.",
        );
        note
    }
}

/// Why run output `output` (its kept output summary) ended short, in plain
/// words for [`RunProgress::note`].
pub fn why_unfinished(output: Option<&str>) -> &'static str {
    // A marker the daemon itself writes ([`STOPPED_BY_RESTART`]).
    if output == Some(STOPPED_BY_RESTART) {
        "a daemon restart stopped it part way."
    } else {
        "it ended before it was done."
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(seq: i64, kind: &str, data: serde_json::Value) -> CliAgentEvent {
        CliAgentEvent {
            seq,
            at: "2026-01-01 00:00:00".into(),
            kind: kind.into(),
            data: data.to_string(),
        }
    }

    #[test]
    fn a_cut_off_run_s_progress_names_its_plan_edits_and_last_steps() {
        let events = vec![
            event(1, "session", serde_json::json!({"id": "s-1"})),
            event(
                2,
                "plan",
                serde_json::json!([
                    {"text": "Read the module", "status": "done"},
                    {"text": "Fix the bug", "status": "in_progress"},
                    {"text": "Add a test", "status": "pending"}
                ]),
            ),
            event(
                3,
                "step",
                serde_json::json!({"id": "a", "words": "Reading lib.rs",
                    "tool": {"kind": "read", "path": "/repo/src/lib.rs"}}),
            ),
            event(
                4,
                "step_ended",
                serde_json::json!({"id": "a", "failed": false}),
            ),
            event(
                5,
                "step",
                serde_json::json!({"id": "b", "words": "Editing lib.rs",
                    "tool": {"kind": "edit", "path": "/repo/src/lib.rs"}}),
            ),
            event(
                6,
                "step_ended",
                serde_json::json!({"id": "b", "failed": false}),
            ),
            event(
                7,
                "step",
                serde_json::json!({"id": "c", "words": "Editing main.rs",
                    "tool": {"kind": "edit", "path": "/repo/src/main.rs"}}),
            ),
            event(
                8,
                "step_ended",
                serde_json::json!({"id": "c", "failed": true}),
            ),
            // Started, never ended: cut off.
            event(
                9,
                "step",
                serde_json::json!({"id": "d", "words": "Editing util.rs",
                    "tool": {"kind": "edit", "path": "/repo/src/util.rs"}}),
            ),
        ];
        let progress = RunProgress::of(&events);
        assert_eq!(progress.edited, vec!["/repo/src/lib.rs".to_string()]);
        assert_eq!(progress.steps_done, 2);
        assert_eq!(
            progress.last_steps,
            vec!["Reading lib.rs", "Editing lib.rs"]
        );
        assert_eq!(progress.plan.len(), 3);

        let note = progress.note("cli-agent-1", why_unfinished(Some(STOPPED_BY_RESTART)));
        assert!(note.contains("a daemon restart stopped it"));
        assert!(note.contains("[done] Read the module"));
        assert!(note.contains("[in progress when it stopped] Fix the bug"));
        assert!(note.contains("[not started] Add a test"));
        assert!(note.contains("/repo/src/lib.rs"));
        assert!(!note.contains("/repo/src/main.rs"));
    }

    #[test]
    fn a_run_with_no_kept_events_has_no_progress() {
        assert!(RunProgress::of(&[]).is_empty());
    }
}
