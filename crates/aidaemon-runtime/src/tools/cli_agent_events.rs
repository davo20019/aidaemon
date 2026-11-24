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

/// One command execution recorded from a coding agent's typed events.
pub(super) struct RecordedRun {
    pub(super) id: Option<String>,
    pub(super) program: Option<String>,
    pub(super) subcommand: Option<String>,
    pub(super) failed: Option<bool>,
}

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
    /// The commands it ran, as its steps named them (`failed == None`: no end seen).
    ran: Vec<RecordedRun>,
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

    /// Read one output line (as the run reads it, after its host), returning
    /// any human-readable step progress labels produced by typed events.
    pub(super) async fn read(&mut self, line: &str) -> Vec<String> {
        let mut progress = Vec::new();
        for event in self.agent.read_events(line) {
            self.structured = true;
            if let Some(words) = self.record(event).await {
                progress.push(words);
            }
        }
        progress
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

    async fn record(&mut self, event: AgentEvent) -> Option<String> {
        let mut step_words = None;
        match &event {
            AgentEvent::Step(step) => {
                if let Some(id) = step.id.as_deref()
                    && !self.steps.insert(id.to_string())
                {
                    return None;
                }
                let words = step.tool.words();
                self.runs.activity(&self.run_id, words.clone());
                step_words = Some(words);
                match &step.tool {
                    aidaemon_coding_agents::StepTool::Edit { path: Some(path) } => {
                        self.runs.edited(&self.run_id, path);
                    }
                    aidaemon_coding_agents::StepTool::Run {
                        program,
                        subcommand,
                    } => {
                        self.ran.push(RecordedRun {
                            id: step.id.clone(),
                            program: program.clone(),
                            subcommand: subcommand.clone(),
                            failed: None,
                        });
                    }
                    _ => {}
                }
            }
            AgentEvent::StepEnded { id, failed } => {
                if let Some(run) = self
                    .ran
                    .iter_mut()
                    .find(|run| run.id.as_deref() == Some(id))
                {
                    run.failed = Some(*failed);
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
            return step_words;
        }
        let (kind, data) = event.to_record();
        keep(&*self.store, self.invocation, &self.seq, kind, data).await;
        step_words
    }
}

/// How long a run's reply is kept: whole, as a client shows it in the run's
/// own view, with a bound only against a runaway agent.
const REPLY_CHARS: usize = 50_000;

/// A run's result: the agent's reply (its report's summary when it reported
/// in fields), its claims, and each claim its recorded actions don't bear
/// out. Files are compared as paths (a claim may be relative to the folder
/// it worked in); a check by its program and optional safe subcommand
/// ([`aidaemon_coding_agents::command_key`], a machine format), judged by
/// the latest matching invocation.
pub(super) fn run_result(
    reported: Option<&(String, Option<aidaemon_coding_agents::Report>)>,
    fallback: &str,
    edited: &[String],
    ran: &[RecordedRun],
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
        let (program, subcommand) = aidaemon_coding_agents::command_key(&check.command);
        let runs: Vec<Option<bool>> = ran
            .iter()
            .filter(|run| {
                run.program.is_some()
                    && run.program == program
                    && match (&subcommand, &run.subcommand) {
                        (Some(claimed), Some(actual)) => claimed == actual,
                        (Some(_), None) => false,
                        (None, _) => true,
                    }
            })
            .map(|run| run.failed)
            .collect();
        if runs.is_empty() {
            mismatches.push(RunMismatch::CheckNotRun {
                command: check.command.clone(),
            });
        } else if runs.last() == Some(&Some(true)) {
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
            RecordedRun {
                id: Some("1".into()),
                program: Some("pytest".into()),
                subcommand: None,
                failed: Some(false),
            },
            RecordedRun {
                id: Some("2".into()),
                program: Some("cargo".into()),
                subcommand: Some("test".into()),
                failed: Some(true),
            },
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

    /// Running `cargo fmt` (passing) never vouches for a claimed `cargo test`,
    /// and a passing `cargo test` followed by a failing `cargo test` reports
    /// failed (the latest invocation decides).
    #[test]
    fn check_verification_distinguishes_subcommands_and_uses_latest_run() {
        let ran = vec![
            RecordedRun {
                id: Some("1".into()),
                program: Some("cargo".into()),
                subcommand: Some("fmt".into()),
                failed: Some(false),
            },
            RecordedRun {
                id: Some("2".into()),
                program: Some("cargo".into()),
                subcommand: Some("check".into()),
                failed: Some(false),
            },
            RecordedRun {
                id: Some("3".into()),
                program: Some("cargo".into()),
                subcommand: Some("check".into()),
                failed: Some(true),
            },
        ];
        let claimed = report(
            &[],
            &[
                ("cargo test", true),
                ("cargo check", true),
                ("cargo fmt", true),
            ],
        );
        let result = run_result(Some(&("{...}".to_string(), Some(claimed))), "", &[], &ran);
        assert_eq!(
            result.mismatches,
            vec![
                RunMismatch::CheckNotRun {
                    command: "cargo test".into()
                },
                RunMismatch::CheckFailed {
                    command: "cargo check".into()
                },
            ]
        );
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
