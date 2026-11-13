//! Work a turn leaves running past its end that is not a helper: a long
//! terminal command or a CLI agent moved to the background.
//!
//! It is a run in [`RunStates`] exactly like a background helper: the
//! turn's card shows it as a typed [`HelperView`] (the same "Running in the
//! background" and "Finished in the background" headers, drawn by
//! `render_card` and kept by `CardEnd::of`), its progress is its latest
//! typed [`Step`], its end is recorded once with the times the runtime
//! measured, and the card settles to its final header before the result is
//! sent, which then never says again how it ended. Nothing here edits a
//! message of its own or counts minutes in text.

use std::sync::{Arc, Weak};

use aidaemon_channels::ports::OutboundRouter;
use aidaemon_channels::turns::TurnOver;
use aidaemon_core::card::Step;
use aidaemon_tools::status::StatusUpdate;
use chrono::{DateTime, Utc};
use tokio::sync::mpsc;
use tracing::warn;

use crate::run_state::{
    GroupEnd, GroupTurn, RunEnd, RunEvidence, RunMeasure, RunReport, RunState, RunStates,
};
use crate::spawn_tool::{
    group_followup, job_end, render_run_result, settle_background_steps, HelperCard,
};

/// The reply of a turn that handed its work to a background run, and the
/// first line of the call that started it: one plain line naming the run.
pub fn started_in_background(label: &str) -> String {
    format!(
        "Started in the background: {}. Its result will come to this chat when it finishes.",
        label.trim()
    )
}

/// The launch line for the runs one turn left running, from their records:
/// one run is named alone ([`started_in_background`]); several are named
/// together, with one answer once the last of them ends.
pub fn started_group_in_background(labels: &[String]) -> String {
    let labels: Vec<&str> = labels
        .iter()
        .map(|label| label.trim())
        .filter(|label| !label.is_empty())
        .collect();
    match labels.as_slice() {
        [] => "Started in the background. Its result will come to this chat when it finishes."
            .to_string(),
        [one] => started_in_background(one),
        [first, second] => format!(
            "Started 2 in the background: {first} and {second}. I'll answer when both finish."
        ),
        [rest @ .., last] => format!(
            "Started {} in the background: {} and {last}. I'll answer when all of them finish.",
            labels.len(),
            rest.join(", ")
        ),
    }
}

/// What a background run's end means for its chat's answer.
#[derive(Debug)]
pub enum RunsTurn {
    /// It is not one of several runs its turn started: it answers alone.
    Alone,
    /// Other runs of its turn still run: its end is on the card, and the
    /// chat is answered once, when the last of them ends.
    Waiting,
    /// It was the last run of its turn's group: the chat gets one answer
    /// from every run's result.
    Answer(GroupAnswer),
}

/// The one answer a turn's group of runs gets.
#[derive(Debug)]
pub struct GroupAnswer {
    /// The continuation the parent answers from ([`group_followup`]).
    pub followup: String,
    /// The turn's task, which the continuation continues.
    pub parent_task_id: Option<String>,
    /// When the owner's request that started the runs came in (the
    /// earliest of their records), how many ran and when the last ended:
    /// the answer's total is measured against it.
    pub clock: Option<crate::run_state::RequestClock>,
    /// Each run's own result message, in start order: what the chat gets
    /// when no answer can be written.
    pub messages: Vec<String>,
}

/// Most characters a run's label keeps.
pub const LABEL_CHARS: usize = 60;

/// The name the owner reads for a run: the title its call gave the work,
/// in plain words, when it is one short line; a path in it reads as the
/// folder or file it names (live 2026-11-13: "Install a Drupal site in
/// /Users/davidloor/projects/claude-drupal-7" was shown as "codex coding
/// agent"). Else `fallback` (what the work is: "a shell loop", "Codex
/// coding agent"). A title is never cut, so a label never ends mid-word or
/// mid-path (live 2026-11-13: "Install a fresh Drupal site … in exactly
/// /Users/davidloor/projects/cla").
pub fn plain_label(title: Option<&str>, fallback: &str) -> String {
    let title = title
        .map(|title| {
            title
                .split_whitespace()
                .map(plain_word)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_default();
    let plain = !title.is_empty() && title.chars().count() <= LABEL_CHARS;
    if plain {
        aidaemon_tools::sanitize::redact_secrets(&title)
    } else {
        fallback.trim().to_string()
    }
}

/// A word of a title as the owner reads it: a path is the name of what it
/// points at.
pub fn plain_word(word: &str) -> String {
    let looks_like_path = word.starts_with('/')
        || word.starts_with('~')
        || word.matches('/').count() > 1
        || word.contains('\\');
    if !looks_like_path {
        return word.to_string();
    }
    let trail: String = word
        .chars()
        .rev()
        .take_while(|c| matches!(c, '.' | ',' | ';' | ':' | ')' | '"' | '\''))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let core = &word[..word.len() - trail.len()];
    let name = core
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty() && *part != "~")
        .unwrap_or(core);
    format!("{name}{trail}")
}

/// Where a background run reports: the runtime's run records, the chat's
/// hub and its checklists.
#[derive(Clone)]
pub struct RunHome {
    pub runs: Arc<RunStates>,
    pub hub: Option<Weak<dyn OutboundRouter>>,
    pub plans: Option<Arc<aidaemon_work::plans::PlanStore>>,
}

/// What starts a background run.
pub struct RunStart<'a> {
    /// Its id everywhere (its run record, the plan steps waiting on it).
    pub id: &'a str,
    /// The chat it reports to.
    pub chat: &'a str,
    pub label: &'a str,
    /// The turn that left it running, whose checklist waits on it.
    pub task_id: Option<&'a str>,
    /// The turn's status channel: its live card shows the run while the
    /// turn lasts.
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    /// When its work began (the command or agent started before it moved to
    /// the background).
    pub work_began: DateTime<Utc>,
}

/// One background run that is not a helper, from its start to its end.
pub struct BackgroundRun {
    home: RunHome,
    id: String,
    chat: String,
    label: String,
    status_tx: Option<mpsc::Sender<StatusUpdate>>,
    /// The turn that started it: its end waits until that turn is over, so
    /// the turn's reply and kept card come first.
    turn: Option<TurnOver>,
    /// The turn's task, when it has one: the run is one of its group.
    task_id: Option<String>,
    work_began: DateTime<Utc>,
    /// Its card once its turn has kept one.
    card: HelperCard,
}

impl BackgroundRun {
    /// Record the run as running and show it on the turn's card. The open
    /// steps of the turn's checklist wait on it, so the plan reads
    /// "continuing in the background" and settles when it ends.
    pub async fn start(home: RunHome, start: RunStart<'_>) -> Arc<Self> {
        home.runs.begin(start.id, start.chat, start.label, None);
        // The runs one turn leaves running are one group, whatever kind they
        // are: the chat is answered once, after the last of them.
        if let Some(task) = start.task_id.filter(|task| !task.trim().is_empty()) {
            home.runs
                .join_group(start.id, &crate::run_state::turn_group(start.chat, task));
        }
        let hub = home.hub.as_ref().and_then(Weak::upgrade);
        let before = match hub.as_ref() {
            Some(hub) => hub.checklist_surface(start.chat).await,
            None => None,
        };
        home.runs.card_before(start.id, before.clone());
        if let Some(plans) = home.plans.as_ref() {
            match plans
                .wait_on_background_jobs(start.chat, start.task_id, &[start.id.to_string()])
                .await
            {
                Ok(Some(plan)) => {
                    if let Some(tx) = start.status_tx.as_ref() {
                        let _ = tx.try_send(StatusUpdate::Checklist(plan.card()));
                    }
                }
                Ok(None) => {}
                Err(error) => warn!(%error, "Could not let the checklist wait on a background run"),
            }
        }
        let run = Arc::new(Self {
            card: HelperCard {
                runs: home.runs.clone(),
                hub: home.hub.clone(),
                plans: home.plans.clone(),
                plan_session: Some(start.chat.to_string()),
                chat: start.chat.to_string(),
                job: start.id.to_string(),
                label: start.label.to_string(),
                at_start: before,
                drawn: Default::default(),
            },
            home,
            id: start.id.to_string(),
            chat: start.chat.to_string(),
            label: start.label.to_string(),
            status_tx: start.status_tx,
            turn: TurnOver::current(),
            task_id: start.task_id.map(str::to_string),
            work_began: start.work_began,
        });
        run.show_on_live_card();
        run
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// The run as its record holds it now.
    pub fn state(&self) -> Option<RunState> {
        self.home.runs.get(&self.id)
    }

    /// Its turn is still running: its live card shows the run.
    fn turn_running(&self) -> bool {
        self.turn.as_ref().is_some_and(|turn| !turn.is_over())
    }

    fn show_on_live_card(&self) {
        let (Some(tx), Some(run)) = (self.status_tx.as_ref(), self.state()) else {
            return;
        };
        if !run.phase.is_terminal() {
            let _ = tx.try_send(StatusUpdate::Helper(run.view()));
        }
    }

    /// Its latest step: on the turn's live card while the turn lasts, else
    /// on the card the turn kept. Ignored once it ended.
    pub async fn step(&self, step: Step) {
        if self
            .state()
            .is_some_and(|run| run.step.as_ref() == Some(&step))
        {
            return;
        }
        if !self.home.runs.step(&self.id, step) {
            return;
        }
        if self.turn_running() {
            self.show_on_live_card();
        } else {
            self.card.refresh().await;
        }
    }

    /// Record how it ended, with the times the runtime measured, once its
    /// turn is over, and settle its card (the plan steps waiting on it and
    /// its own final header). Returns its record and whether a card now
    /// shows how it ended. A second end changes nothing.
    pub async fn finish(&self, end: RunEnd, evidence: RunEvidence) -> (Option<RunState>, bool) {
        if let Some(turn) = self.turn.as_ref() {
            turn.wait().await;
        }
        let phase = end.phase();
        let ended = self.home.runs.finish(
            &self.id,
            end,
            RunMeasure {
                ended_at: Utc::now(),
                work_began: Some(self.work_began),
                waited: std::time::Duration::ZERO,
            },
            evidence,
            RunReport {
                text: String::new(),
                untrusted: false,
            },
        );
        let Some(ended) = ended else {
            return (self.state(), false);
        };
        let on_card = settle_background_steps(
            &self.home.runs,
            self.home.plans.as_ref(),
            self.home.hub.as_ref(),
            Some(&self.chat),
            &self.chat,
            &self.id,
            job_end(phase),
        )
        .await;
        (Some(ended), on_card)
    }

    /// Its end within its turn's group (`state`, from [`Self::finish`]):
    /// `report` is its result as the parent reads it, `message` what the
    /// chat gets for it alone. Given once per run.
    pub fn end_in_group(
        &self,
        state: &RunState,
        report: RunReport,
        message: String,
        max_report_chars: usize,
    ) -> RunsTurn {
        let end = GroupEnd {
            state: state.clone(),
            report,
            message,
        };
        match self.home.runs.group_end(&self.id, end) {
            GroupTurn::Alone => RunsTurn::Alone,
            GroupTurn::Waiting => RunsTurn::Waiting,
            GroupTurn::Complete(ends) => RunsTurn::Answer(GroupAnswer {
                followup: group_followup(&ends, max_report_chars),
                parent_task_id: self.task_id.clone(),
                clock: crate::run_state::RequestClock::of_group(&ends),
                messages: ends.into_iter().map(|end| end.message).collect(),
            }),
        }
    }

    /// Its result for the chat, from its record: the body, how it ended
    /// only when no card shows it, and the time the runtime measured.
    pub fn result(&self, state: Option<&RunState>, body: &str, on_card: bool) -> String {
        match state {
            Some(state) => render_run_result(state, &self.label, body, body.len().max(1), on_card),
            None => body.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_label_is_the_plain_title_or_what_the_work_is() {
        assert_eq!(
            plain_label(Some("  install the\nDrupal site "), "a shell loop"),
            "install the Drupal site"
        );
        assert_eq!(
            plain_label(
                Some("Install a fresh Drupal site in exactly /Users/someone/projects/claude-test"),
                "Codex coding agent"
            ),
            "Install a fresh Drupal site in exactly claude-test"
        );
        assert_eq!(
            plain_label(Some("Fix ~/src/app/main.rs."), "a shell loop"),
            "Fix main.rs."
        );
        assert_eq!(
            plain_label(Some(&"word ".repeat(30)), "a shell loop"),
            "a shell loop"
        );
        assert_eq!(plain_label(None, "ddev"), "ddev");
    }

    #[test]
    fn the_launch_line_names_every_run_of_the_turn() {
        assert_eq!(
            started_group_in_background(&["counting script".into()]),
            started_in_background("counting script")
        );
        assert_eq!(
            started_group_in_background(&["Install Drupal".into(), "Install WordPress".into()]),
            "Started 2 in the background: Install Drupal and Install WordPress. I'll answer \
             when both finish."
        );
        assert_eq!(
            started_group_in_background(&["a".into(), "b".into(), "c".into()]),
            "Started 3 in the background: a, b and c. I'll answer when all of them finish."
        );
    }

    /// A command and a CLI agent one turn left running are one group: the
    /// first end waits, the last one answers the turn once from both.
    #[tokio::test]
    async fn the_runs_one_turn_leaves_running_answer_once() {
        let home = RunHome {
            runs: Arc::new(RunStates::default()),
            hub: None,
            plans: None,
        };
        let start = |id: &'static str, label: &'static str| RunStart {
            id,
            chat: "chat",
            label,
            task_id: Some("task-1"),
            status_tx: None,
            work_began: Utc::now(),
        };
        let command = BackgroundRun::start(home.clone(), start("command-1", "count")).await;
        let agent = BackgroundRun::start(home.clone(), start("cli-agent-1", "Install")).await;
        let end = |run: &BackgroundRun, state: Option<RunState>, text: &str| {
            let state = state.expect("its record");
            run.end_in_group(
                &state,
                RunReport {
                    text: text.to_string(),
                    untrusted: false,
                },
                format!("{} done", run.label()),
                1000,
            )
        };
        let (state, _) = command
            .finish(RunEnd::Succeeded, RunEvidence::default())
            .await;
        assert!(matches!(end(&command, state, "8"), RunsTurn::Waiting));
        let (state, _) = agent
            .finish(RunEnd::Succeeded, RunEvidence::default())
            .await;
        match end(&agent, state, "installed") {
            RunsTurn::Answer(answer) => {
                assert_eq!(answer.parent_task_id.as_deref(), Some("task-1"));
                assert_eq!(answer.messages, vec!["count done", "Install done"]);
                assert!(answer.followup.contains("installed"));
            }
            other => panic!("expected the group's answer, got {other:?}"),
        }

        // A run alone in its turn answers alone.
        let solo = BackgroundRun::start(
            home,
            RunStart {
                task_id: Some("task-2"),
                ..start("command-2", "solo")
            },
        )
        .await;
        let (state, _) = solo.finish(RunEnd::Succeeded, RunEvidence::default()).await;
        assert!(matches!(end(&solo, state, "ok"), RunsTurn::Alone));
    }

    #[test]
    fn the_handoff_line_names_the_run() {
        assert_eq!(
            started_in_background("counting script"),
            "Started in the background: counting script. Its result will come to this chat \
             when it finishes."
        );
    }
}
