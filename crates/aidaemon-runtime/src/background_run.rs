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

use crate::run_state::{RunEvidence, RunMeasure, RunPhase, RunReport, RunState, RunStates};
use crate::spawn_tool::{job_end, render_run_result, settle_background_steps, HelperCard};

/// The reply of a turn that handed its work to a background run, and the
/// first line of the call that started it: one plain line naming the run.
pub fn started_in_background(label: &str) -> String {
    format!(
        "Started in the background: {}. Its result will come to this chat when it finishes.",
        label.trim()
    )
}

/// Most characters a run's label keeps.
pub const LABEL_CHARS: usize = 60;

/// The name the owner reads for a run: the title its call gave the work,
/// in plain words, when it is one short line with no path in it; else
/// `fallback` (what the work is: "a shell loop", "Codex coding agent"). A
/// title is never cut, so a label never ends mid-word or mid-path (live
/// 2026-11-13: "Install a fresh Drupal site … in exactly
/// /Users/davidloor/projects/cla").
pub fn plain_label(title: Option<&str>, fallback: &str) -> String {
    let title = title
        .map(|title| title.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_default();
    let plain = !title.is_empty()
        && title.chars().count() <= LABEL_CHARS
        && !title
            .split_whitespace()
            .any(|word| word.contains('/') || word.contains('\\') || word.starts_with('~'));
    if plain {
        aidaemon_tools::sanitize::redact_secrets(&title)
    } else {
        fallback.trim().to_string()
    }
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
    pub async fn finish(&self, phase: RunPhase, evidence: RunEvidence) -> (Option<RunState>, bool) {
        if let Some(turn) = self.turn.as_ref() {
            turn.wait().await;
        }
        let ended = self.home.runs.finish(
            &self.id,
            phase,
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
            "Codex coding agent"
        );
        assert_eq!(
            plain_label(Some(&"word ".repeat(30)), "a shell loop"),
            "a shell loop"
        );
        assert_eq!(plain_label(None, "ddev"), "ddev");
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
