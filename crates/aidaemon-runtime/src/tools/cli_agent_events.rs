//! A launched coding agent's typed events, kept as it runs.
//!
//! Each output line goes through the agent's own reader
//! ([`aidaemon_coding_agents::CodingAgent::read_events`]); each event is kept
//! on the run's invocation in order, its session id on the invocation
//! itself, and its latest step becomes the run record's activity, which the
//! owner's app shows while it runs.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use aidaemon_coding_agents::{ActionJudge, AgentAction, AgentEvent, CodingAgent, Verdict};
use async_trait::async_trait;
use serde_json::json;

use super::cli_agent::CliAgentStore;
use crate::run_state::RunStates;

pub(super) struct AgentEventRecorder {
    agent: Arc<dyn CodingAgent>,
    store: Arc<dyn CliAgentStore>,
    /// The invocation row the events are kept on (0: none was logged).
    invocation: i64,
    runs: Arc<RunStates>,
    run_id: String,
    /// The events' order, shared with the run's [`RecordingJudge`].
    seq: Arc<AtomicI64>,
    /// Steps already recorded, by the agent's own id: an item reported
    /// started and again completed is one step.
    steps: HashSet<String>,
    /// Its output read as typed events at least once: its steps come from
    /// them, not from how much it printed.
    structured: bool,
    /// The commands it ran, as its steps named them: each program, by the
    /// step's id, and whether its step ended failed (`None`: no end seen).
    ran: Vec<(Option<String>, Option<String>, Option<bool>)>,
    /// What it ended its run with ([`AgentEvent::Reported`]).
    reported: Option<(String, Option<aidaemon_coding_agents::Report>)>,
}

impl AgentEventRecorder {
    pub(super) fn new(
        agent: Arc<dyn CodingAgent>,
        store: Arc<dyn CliAgentStore>,
        invocation: i64,
        runs: Arc<RunStates>,
        run_id: String,
    ) -> Self {
        Self {
            agent,
            store,
            invocation,
            runs,
            run_id,
            seq: Arc::default(),
            steps: HashSet::new(),
            structured: false,
            ran: Vec::new(),
            reported: None,
        }
    }

    /// `judge`, keeping each decision it makes as one of this run's events.
    pub(super) fn judging(&self, judge: Arc<dyn ActionJudge>) -> Arc<dyn ActionJudge> {
        Arc::new(RecordingJudge {
            judge,
            store: self.store.clone(),
            invocation: self.invocation,
            seq: self.seq.clone(),
        })
    }

    /// Read one output line (as the run reads it, after its host).
    pub(super) async fn read(&mut self, line: &str) {
        for event in self.agent.read_events(line) {
            self.structured = true;
            self.record(event).await;
        }
    }

    /// What its coding agent ended with, checked against what it was seen
    /// doing: `edited`, every file its run changed (its steps' and its
    /// folders' own record). `fallback`: its output, for an agent that
    /// reported nothing typed (a plain program); a structured agent that
    /// reported nothing has no reply.
    pub(super) fn result(
        &self,
        edited: &[String],
        fallback: &str,
    ) -> aidaemon_channels::turns::RunResult {
        let fallback = if self.structured() { "" } else { fallback };
        run_result(self.reported.as_ref(), fallback, edited, &self.ran)
    }

    /// What its agent ended with, as its adapter read it: a report's text
    /// ([`aidaemon_coding_agents::Report::to_text`]) or its final message.
    /// `None` when it reported nothing typed (a plain program).
    pub(super) fn reported_text(&self) -> Option<String> {
        let (text, report) = self.reported.as_ref()?;
        let text = match report {
            Some(report) => report.to_text(),
            None => text.trim().to_string(),
        };
        (!text.is_empty()).then_some(text)
    }

    /// Whether its output reads as typed events ([`Self::read`] found one).
    pub(super) fn structured(&self) -> bool {
        self.structured || self.agent.reports_steps()
    }

    async fn record(&mut self, event: AgentEvent) {
        match &event {
            AgentEvent::Step(step) => {
                if let Some(id) = step.id.as_deref()
                    && !self.steps.insert(id.to_string())
                {
                    return;
                }
                self.runs.activity(&self.run_id, step.tool.words());
                match &step.tool {
                    aidaemon_coding_agents::StepTool::Edit { path: Some(path) } => {
                        self.runs.edited(&self.run_id, path);
                    }
                    aidaemon_coding_agents::StepTool::Run { program } => {
                        self.ran.push((step.id.clone(), program.clone(), None));
                    }
                    _ => {}
                }
            }
            AgentEvent::StepEnded { id, failed } => {
                if let Some(run) = self.ran.iter_mut().find(|run| run.0.as_deref() == Some(id)) {
                    run.2 = Some(*failed);
                }
            }
            AgentEvent::Reported { text, report } => {
                self.reported = Some((text.clone(), report.clone()));
            }
            AgentEvent::Session { id, resumable, .. } if self.invocation != 0 => {
                if let Err(error) = self.store.set_cli_agent_session(self.invocation, id).await {
                    tracing::warn!(%error, "Could not keep a coding agent's session id");
                }
                // The agent said, as it ran, whether its session can be
                // continued (its launch could not tell).
                if let Some(kept) = *resumable {
                    self.runs.keeps_session(&self.run_id, kept);
                }
                if let Some(kept) = *resumable
                    && let Err(error) = self
                        .store
                        .set_cli_agent_keeps_session(self.invocation, kept)
                        .await
                {
                    tracing::warn!(%error, "Could not note whether a coding agent kept its session");
                }
            }
            _ => {}
        }
        if self.invocation == 0 {
            return;
        }
        let (kind, data) = event.to_record();
        keep(&*self.store, self.invocation, &self.seq, kind, data).await;
    }
}

/// How long a run's reply is kept: whole, as a client shows it in the run's
/// own view, with a bound only against a runaway agent.
const REPLY_CHARS: usize = 50_000;

/// A run's result: the agent's reply (its report's summary when it reported
/// in fields), its claims, and each claim its recorded actions don't bear
/// out. Files are compared as paths (a claim may be relative to the folder
/// it worked in); a check by the program it runs, as its steps named
/// programs ([`aidaemon_coding_agents::program_of`], a machine format).
pub(super) fn run_result(
    reported: Option<&(String, Option<aidaemon_coding_agents::Report>)>,
    fallback: &str,
    edited: &[String],
    ran: &[(Option<String>, Option<String>, Option<bool>)],
) -> aidaemon_channels::turns::RunResult {
    use aidaemon_channels::turns::{RunClaims, RunMismatch, RunResult};
    let shorten = |text: &str| -> String {
        let text = text.trim();
        match text.char_indices().nth(REPLY_CHARS) {
            Some((at, _)) => format!("{}…", &text[..at]),
            None => text.to_string(),
        }
    };
    let Some((text, report)) = reported else {
        return RunResult {
            text: shorten(fallback),
            ..RunResult::default()
        };
    };
    let Some(report) = report else {
        return RunResult {
            text: shorten(text),
            ..RunResult::default()
        };
    };
    // A path names the same file when one is the other, or ends with the
    // other as a relative path.
    let same = |a: &str, b: &str| {
        let (a, b) = (a.trim_start_matches("./"), b.trim_start_matches("./"));
        a == b || a.ends_with(&format!("/{b}")) || b.ends_with(&format!("/{a}"))
    };
    let mut mismatches = Vec::new();
    for path in &report.changed_files {
        if !edited.iter().any(|seen| same(seen, path)) {
            mismatches.push(RunMismatch::ChangeNotSeen { path: path.clone() });
        }
    }
    for path in edited {
        if !report
            .changed_files
            .iter()
            .any(|claimed| same(path, claimed))
        {
            mismatches.push(RunMismatch::ChangeNotClaimed { path: path.clone() });
        }
    }
    for check in report.checks.iter().filter(|check| check.passed) {
        let program = aidaemon_coding_agents::program_of(&check.command);
        let runs: Vec<Option<bool>> = ran
            .iter()
            .filter(|(_, ran, _)| ran.is_some() && *ran == program)
            .map(|(_, _, failed)| *failed)
            .collect();
        if runs.is_empty() {
            mismatches.push(RunMismatch::CheckNotRun {
                command: check.command.clone(),
            });
        } else if runs.iter().all(|failed| *failed == Some(true)) {
            mismatches.push(RunMismatch::CheckFailed {
                command: check.command.clone(),
            });
        }
    }
    RunResult {
        text: shorten(&report.summary),
        claims: Some(RunClaims {
            summary: report.summary.clone(),
            changed_files: report.changed_files.clone(),
            checks: report
                .checks
                .iter()
                .map(|check| (check.command.clone(), check.passed))
                .collect(),
            open_questions: report.open_questions.clone(),
            needs_owner: report.needs_owner,
        }),
        mismatches,
    }
}

async fn keep(
    store: &dyn CliAgentStore,
    invocation: i64,
    seq: &AtomicI64,
    kind: &str,
    data: serde_json::Value,
) {
    let seq = seq.fetch_add(1, Ordering::Relaxed) + 1;
    if let Err(error) = store
        .record_cli_agent_event(invocation, seq, kind, &data.to_string())
        .await
    {
        tracing::warn!(%error, "Could not keep a coding agent's event");
    }
}

/// The runtime's judge for a hosted agent's requests, keeping each
/// decision as a `permission` event: what was asked (its kind and the
/// owner's label for it), whether it was allowed, and why not.
struct RecordingJudge {
    judge: Arc<dyn ActionJudge>,
    store: Arc<dyn CliAgentStore>,
    invocation: i64,
    seq: Arc<AtomicI64>,
}

#[async_trait]
impl ActionJudge for RecordingJudge {
    async fn judge(&self, action: AgentAction) -> Verdict {
        let kind = match &action {
            AgentAction::RunCommand { .. } => "run",
            AgentAction::WriteFile { .. } => "write",
            AgentAction::EditFile { .. } => "edit",
            AgentAction::ReadFile { .. } => "read",
            AgentAction::Search { .. } => "search",
            AgentAction::Fetch { .. } => "fetch",
            AgentAction::ReachHost { .. } => "network",
        };
        let label = action.label();
        let verdict = self.judge.judge(action).await;
        if self.invocation != 0 {
            let (outcome, reason) = match &verdict {
                Verdict::Allow => ("allowed", None),
                Verdict::Deny { reason } => ("refused", Some(reason.clone())),
                Verdict::Declined { reason } => ("declined", Some(reason.clone())),
                Verdict::Unanswered { reason } => ("unanswered", Some(reason.clone())),
                // A verdict added later: not allowed, with its own reason.
                other => ("refused", other.reason().map(str::to_string)),
            };
            keep(
                &*self.store,
                self.invocation,
                &self.seq,
                "permission",
                json!({"action": kind, "label": label, "outcome": outcome, "reason": reason}),
            )
            .await;
        }
        verdict
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_channels::turns::RunMismatch;
    use aidaemon_coding_agents::{Check, Report};

    fn report(changed: &[&str], checks: &[(&str, bool)]) -> Report {
        Report {
            summary: "Added the slug test".into(),
            changed_files: changed.iter().map(|path| path.to_string()).collect(),
            checks: checks
                .iter()
                .map(|(command, passed)| Check {
                    command: command.to_string(),
                    passed: *passed,
                })
                .collect(),
            open_questions: Vec::new(),
            needs_owner: false,
        }
    }

    /// Each claim is checked against what was recorded; the record wins.
    #[test]
    fn a_report_s_claims_are_checked_against_the_record() {
        let edited = vec![
            "/w/synthetic/slug.py".to_string(),
            "/w/synthetic/notes.md".to_string(),
        ];
        let ran = vec![
            (Some("1".into()), Some("pytest".into()), Some(false)),
            (Some("2".into()), Some("cargo".into()), Some(true)),
        ];
        let claimed = report(
            &["slug.py", "tests/test_slug.py"],
            &[
                ("pytest -q", true),
                ("cargo test", true),
                ("ruff check .", true),
            ],
        );
        let result = run_result(
            Some(&("{...}".to_string(), Some(claimed))),
            "",
            &edited,
            &ran,
        );
        assert_eq!(
            result.text, "Added the slug test",
            "its summary is its reply"
        );
        assert_eq!(
            result.mismatches,
            vec![
                RunMismatch::ChangeNotSeen {
                    path: "tests/test_slug.py".into()
                },
                RunMismatch::ChangeNotClaimed {
                    path: "/w/synthetic/notes.md".into()
                },
                RunMismatch::CheckFailed {
                    command: "cargo test".into()
                },
                RunMismatch::CheckNotRun {
                    command: "ruff check .".into()
                },
            ]
        );
        assert!(result.claims.is_some());
    }

    /// Text alone: no claims, nothing to check. A plain program's output
    /// stands for its reply.
    #[test]
    fn a_reply_without_a_report_has_no_claims() {
        let said = ("It has 4 lines.".to_string(), None);
        let result = run_result(Some(&said), "", &[], &[]);
        assert_eq!(result.text, "It has 4 lines.");
        assert!(result.claims.is_none() && result.mismatches.is_empty());
        let plain = run_result(None, "  printed output ", &[], &[]);
        assert_eq!(plain.text, "printed output");
    }
}
