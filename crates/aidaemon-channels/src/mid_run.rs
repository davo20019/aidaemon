//! Messages that arrive while a run is going.
//!
//! The [`crate::turns::TurnRunner`] keeps each session's running turn's own
//! latest step ([`LiveWork`], fed by the turn's status updates). Its
//! helpers and plan card are read from the runtime's records
//! ([`crate::ports::ChannelAgentRuntime::chat_runs`]), the same views the
//! card draws, keyed by run id. When the owner writes mid-run, the agent
//! reads the message
//! ([`crate::ports::ChannelAgentRuntime::read_mid_run_message`]): a question
//! about the work is answered at once from that state, without touching the
//! run; a change to the work joins it; anything else waits for its own turn.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aidaemon_core::card::RunPhase;
use aidaemon_tools::duration_format::{ZeroUnitStyle, compact_seconds};
use aidaemon_tools::status::StatusUpdate;

use crate::ports::{ChatRun, ChatWorkKind, RunningWork};

/// How many helpers a reply lists.
const MAX_HELPERS: usize = 6;

#[derive(Debug, Default)]
struct Progress {
    task_id: u64,
    last_step: Option<(String, Instant)>,
}

/// Each session's running turn's own latest step, as its status updates
/// showed it. Helpers and the plan card are not kept here: they are read
/// from the runtime's records, so a reply never keeps a second account of
/// them.
#[derive(Clone, Default)]
pub struct LiveWork {
    sessions: Arc<Mutex<HashMap<String, Progress>>>,
}

impl LiveWork {
    /// Start tracking `task_id`, the session's new turn.
    pub fn begin(&self, session_id: &str, task_id: u64) {
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.insert(
                session_id.to_string(),
                Progress {
                    task_id,
                    ..Progress::default()
                },
            );
        }
    }

    /// Stop tracking `task_id` (its turn ended).
    pub fn end(&self, session_id: &str, task_id: u64) {
        if let Ok(mut sessions) = self.sessions.lock()
            && sessions
                .get(session_id)
                .is_some_and(|p| p.task_id == task_id)
        {
            sessions.remove(session_id);
        }
    }

    /// Note what `update` says about `task_id`'s own work.
    pub fn record(&self, session_id: &str, task_id: u64, update: &StatusUpdate) {
        let now = Instant::now();
        let Ok(mut sessions) = self.sessions.lock() else {
            return;
        };
        let Some(progress) = sessions
            .get_mut(session_id)
            .filter(|progress| progress.task_id == task_id)
        else {
            return;
        };
        let step = match update {
            StatusUpdate::ToolStart { kind, .. } => {
                kind.says_what().then(|| kind.words().to_string())
            }
            StatusUpdate::ProgressSummary { summary, .. } => Some(summary.clone()),
            StatusUpdate::PlanStepStart {
                step_index,
                total_steps,
                description,
                ..
            } => Some(format!(
                "step {} of {total_steps}: {description}",
                step_index + 1
            )),
            _ => None,
        };
        if let Some(step) = step.filter(|step| !step.trim().is_empty()) {
            progress.last_step = Some((step, now));
        }
    }

    /// The session's running work: `request` started `elapsed` ago (empty
    /// when no turn runs), its own latest step, and the chat's helper runs
    /// and plan card as the runtime's records show them (`runs`).
    pub fn snapshot(
        &self,
        session_id: &str,
        request: &str,
        elapsed: Duration,
        runs: crate::ports::ChatRuns,
    ) -> RunningWork {
        let mut work = RunningWork {
            request: request.to_string(),
            elapsed,
            plan: runs.plan,
            runs: runs.runs,
            ..RunningWork::default()
        };
        if let Ok(sessions) = self.sessions.lock()
            && let Some(progress) = sessions.get(session_id)
        {
            work.last_step = progress
                .last_step
                .as_ref()
                .map(|(step, at)| (step.clone(), at.elapsed()));
        }
        work
    }
}

fn ago(duration: Duration) -> String {
    compact_seconds(
        i64::try_from(duration.as_secs()).unwrap_or(i64::MAX),
        ZeroUnitStyle::Trim,
    )
}

/// A helper run's words, as its line on the card says them: its latest
/// step while it runs, how it ended once it ended, with the reason its
/// record gives.
fn run_words(run: &ChatRun) -> String {
    let words = crate::live_status::helper_status_words(&run.view);
    match run.view.reason.as_deref().map(str::trim) {
        Some(reason) if run.view.phase.is_terminal() && !reason.is_empty() => {
            format!("{words}. {reason}")
        }
        _ => words,
    }
}

/// Where the running work stands, from its typed state: the request and how
/// long it has run, its latest step, then one line per helper run (from its
/// record, as the card draws it) and background command, with the time
/// from the running list, the same clock and numbers `/tasks` shows. A
/// helper is matched to its running item by its run id. A helper of the
/// running turn that ended says how it ended, as its card does; one an
/// earlier turn started that ended is left out.
pub fn running_work_status(work: &RunningWork) -> String {
    let mut lines = vec![if work.request.trim().is_empty() {
        // No turn runs: only helpers and background commands do.
        "Working in the background:".to_string()
    } else {
        format!(
            "Still working on \"{}\" ({} so far).",
            crate::presentation::word_preview(&work.request, 80),
            ago(work.elapsed)
        )
    }];
    if let Some((step, since)) = &work.last_step {
        lines.push(format!(
            "Latest step: {} ({} ago).",
            crate::presentation::word_preview(step, 120),
            ago(*since)
        ));
    }
    let mut items = Vec::new();
    let mut listed_runs: Vec<&str> = Vec::new();
    for item in work.running.iter().filter(|item| item.work.is_some()) {
        let run = (item.work == Some(ChatWorkKind::SubAgent))
            .then(|| {
                let id = item.run.as_deref()?;
                work.runs.iter().find(|run| run.id == id)
            })
            .flatten();
        items.push(match run {
            Some(run) => {
                listed_runs.push(&run.id);
                format!(
                    "• {}: {} ({} so far)",
                    crate::presentation::word_preview(&run.view.label, 60),
                    crate::presentation::word_preview(&run_words(run), 80),
                    ago(item.elapsed)
                )
            }
            None => format!(
                "• {} ({} so far)",
                crate::presentation::word_preview(&item.title, 80),
                ago(item.elapsed)
            ),
        });
    }
    for run in work
        .runs
        .iter()
        .filter(|run| !listed_runs.contains(&run.id.as_str()))
        .filter(|run| run.this_request || run.view.phase == RunPhase::Running)
    {
        items.push(format!(
            "• {}: {}",
            crate::presentation::word_preview(&run.view.label, 60),
            crate::presentation::word_preview(&run_words(run), 80),
        ));
    }
    lines.extend(items.into_iter().take(MAX_HELPERS));
    lines.join("\n")
}

/// The reply to a question about the running work: what its typed state
/// says, in plain words.
pub fn running_work_reply(work: &RunningWork) -> String {
    let mut lines = vec![running_work_status(work)];
    if let Some(plan) = &work.plan {
        lines.push(String::new());
        lines.push(plan.trim().to_string());
    }
    lines.push(String::new());
    lines.push("I'll send the results when it's done.".to_string());
    lines.join("\n")
}

/// The notice for a message the agent could not read (its model call
/// failed): the message joins the running work, and the owner sees where
/// that work stands, so a question about it is answered either way.
pub fn unread_message_notice(work: &RunningWork) -> String {
    let notice = crate::presentation::queued_message_notice(false);
    if work.request.trim().is_empty() {
        return notice;
    }
    format!("{notice}\n\n{}", running_work_status(work))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::card::{ActivityKind, HelperView, Step};

    fn run(id: &str, view: HelperView, this_request: bool) -> ChatRun {
        ChatRun {
            id: id.into(),
            view,
            running_for: Duration::from_secs(10),
            this_request,
        }
    }

    fn ended(label: &str, phase: RunPhase) -> HelperView {
        HelperView {
            run: None,
            phase,
            ..HelperView::running(label, false, None)
        }
    }

    #[test]
    fn the_state_follows_its_own_turn_only() {
        let live = LiveWork::default();
        live.begin("s", 7);
        live.record(
            "s",
            7,
            &StatusUpdate::ToolStart {
                name: "terminal".into(),
                summary: "installing the site".into(),
                kind: ActivityKind::Run,
            },
        );
        // A late update from an earlier turn changes nothing.
        live.record(
            "s",
            6,
            &StatusUpdate::ToolStart {
                name: "terminal".into(),
                summary: "old step".into(),
                kind: ActivityKind::Web,
            },
        );
        let records = crate::ports::ChatRuns {
            runs: vec![run(
                "helper:1",
                HelperView::running(
                    "Set up the blog",
                    true,
                    Some(Step::Doing(ActivityKind::Write)),
                ),
                true,
            )],
            plan: Some("☑ one\n☐ two".into()),
        };
        let work = live.snapshot(
            "s",
            "Set up two demo sites",
            Duration::from_secs(300),
            records,
        );
        assert_eq!(work.last_step.as_ref().unwrap().0, "Running a command");
        assert_eq!(work.plan.as_deref(), Some("☑ one\n☐ two"));
        assert_eq!(work.runs.len(), 1);

        let none = crate::ports::ChatRuns::default();
        live.end("s", 6);
        assert!(
            live.snapshot("s", "x", Duration::ZERO, none.clone())
                .last_step
                .is_some()
        );
        live.end("s", 7);
        assert!(
            live.snapshot("s", "x", Duration::ZERO, none)
                .last_step
                .is_none()
        );
    }

    /// Live 2026-11-13: "how's it going?" got one paragraph ("about 38
    /// seconds... each about 10 seconds") while `/tasks` said 55s and 26s.
    /// The reply is the typed state, one line per helper with its current
    /// step and the running list's time: the numbers `/tasks` shows.
    #[test]
    fn the_status_reply_lists_each_helper_with_the_times_tasks_shows() {
        let now = chrono::Utc::now();
        let list = crate::stop::WorkList::for_tests(vec![
            (
                1,
                "Sub-agent: Slow count A",
                now - chrono::Duration::seconds(26),
                "r",
            ),
            (
                2,
                "Sub-agent: Slow count B",
                now - chrono::Duration::seconds(26),
                "r",
            ),
        ]);
        let work = RunningWork {
            request: "Count to 100 twice, slowly".into(),
            elapsed: Duration::from_secs(55),
            last_step: None,
            plan: None,
            runs: [("run:1", "Slow count A"), ("run:2", "Slow count B")]
                .into_iter()
                .map(|(id, label)| {
                    run(
                        id,
                        HelperView::running(label, false, Some(Step::Doing(ActivityKind::Run))),
                        true,
                    )
                })
                .collect(),
            running: list.items(now),
        };
        let reply = running_work_reply(&work);
        let tasks = list.lines(now);
        assert_eq!(tasks[0], "#1 Sub-agent: Slow count A (26s so far)");
        assert!(reply.starts_with("Still working on \"Count to 100 twice, slowly\" (55s so far)."));
        assert!(
            reply.contains("\n• Slow count A: Running a command (26s so far)\n"),
            "{reply}"
        );
        assert!(
            reply.contains("\n• Slow count B: Running a command (26s so far)"),
            "{reply}"
        );
        assert!(!reply.contains("10s"), "{reply}");
        assert!(reply.ends_with("I'll send the results when it's done."));
    }

    /// Live 2026-11-14: "how is it going?" said the Drupal helper stopped
    /// while the card said it finished. Two attempts with the same label
    /// were one entry in a second, label-keyed account of the helpers. The
    /// reply now reads the runs' records by id, as the card does: two
    /// attempts stay two lines, each saying what its card line says.
    #[test]
    fn two_attempts_with_one_label_stay_two_and_agree_with_the_card() {
        let label = "Install Drupal";
        let mut first = ended(label, RunPhase::Stopped);
        first.reason = Some("It ran out of time.".into());
        let second = ended(label, RunPhase::Succeeded);
        let work = RunningWork {
            request: "Set up the Drupal site".into(),
            elapsed: Duration::from_secs(120),
            runs: vec![
                run("helper:a", first.clone(), true),
                run("helper:b", second.clone(), true),
            ],
            ..RunningWork::default()
        };
        let reply = running_work_status(&work);
        let lines: Vec<&str> = reply.lines().filter(|line| line.starts_with('•')).collect();
        assert_eq!(
            lines,
            vec![
                "• Install Drupal: Didn't finish. It ran out of time.",
                "• Install Drupal: Finished",
            ],
            "{reply}"
        );

        // The card, drawn from the same records, says the same of each.
        let mut card = crate::live_status::LiveStatus::new();
        card.note(&StatusUpdate::Helper(first.clone()));
        card.note(&StatusUpdate::Helper(HelperView::running(
            label, false, None,
        )));
        card.note(&StatusUpdate::Helper(second.clone()));
        let drawn = card.preview().unwrap();
        assert!(drawn.contains(first.headline()), "{drawn}");
        assert!(drawn.contains(second.headline()), "{drawn}");
        assert!(drawn.contains("It ran out of time."), "{drawn}");
        for view in [&first, &second] {
            assert!(reply.contains(&crate::live_status::helper_status_words(view)));
        }
    }

    /// A helper an earlier turn started that has ended is history: the
    /// reply leaves it out; one still running is listed.
    #[test]
    fn an_earlier_turns_ended_helper_is_left_out() {
        let work = RunningWork {
            request: "Next thing".into(),
            runs: vec![
                run("helper:old", ended("Old job", RunPhase::Succeeded), false),
                run(
                    "helper:bg",
                    HelperView::running("Background job", true, None),
                    false,
                ),
            ],
            ..RunningWork::default()
        };
        let reply = running_work_status(&work);
        assert!(!reply.contains("Old job"), "{reply}");
        assert!(
            reply.contains("• Background job: Getting started"),
            "{reply}"
        );
    }
}
