//! Messages that arrive while a run is going.
//!
//! The [`crate::turns::TurnRunner`] keeps each session's running work as
//! typed live state ([`LiveWork`], fed by the run's status updates). When
//! the owner writes mid-run, the agent reads the message
//! ([`crate::ports::ChannelAgentRuntime::read_mid_run_message`]): a question
//! about the work is answered at once from that state, without touching the
//! run; a change to the work joins it; anything else waits for its own turn.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aidaemon_tools::duration_format::{compact_seconds, ZeroUnitStyle};
use aidaemon_tools::status::StatusUpdate;

use crate::ports::{ChatWorkKind, HelperProgress, RunningWork};

/// How many helpers a reply lists.
const MAX_HELPERS: usize = 6;

/// How many helpers a session keeps the last step of.
const MAX_TRACKED_HELPERS: usize = 32;

#[derive(Debug, Default)]
struct Progress {
    task_id: u64,
    last_step: Option<(String, Instant)>,
    plan: Option<String>,
}

#[derive(Debug)]
struct Helper {
    /// The turn that started it.
    task_id: u64,
    label: String,
    progress: String,
    at: Instant,
    started: Instant,
}

/// Each session's running work as its status updates showed it: the
/// running turn's own steps and plan, and every helper's latest step. A
/// helper outlives the turn that started it (it runs in the background), so
/// its steps are kept for the session, not the turn: a question asked while
/// only helpers run is answered with where each one stands.
#[derive(Clone, Default)]
pub struct LiveWork {
    sessions: Arc<Mutex<HashMap<String, Progress>>>,
    helpers: Arc<Mutex<HashMap<String, Vec<Helper>>>>,
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

    /// Stop tracking `task_id` (its turn ended). Its helpers' steps stay.
    pub fn end(&self, session_id: &str, task_id: u64) {
        if let Ok(mut sessions) = self.sessions.lock() {
            if sessions
                .get(session_id)
                .is_some_and(|p| p.task_id == task_id)
            {
                sessions.remove(session_id);
            }
        }
    }

    /// Note what `update` says about `task_id`'s work. A helper's start
    /// and steps are noted whichever turn started it.
    pub fn record(&self, session_id: &str, task_id: u64, update: &StatusUpdate) {
        let now = Instant::now();
        if let StatusUpdate::Helper(view) = update {
            let line = crate::live_status::helper_status_words(view);
            let Ok(mut helpers) = self.helpers.lock() else {
                return;
            };
            let helpers = helpers.entry(session_id.to_string()).or_default();
            match helpers.iter_mut().find(|h| h.label == view.label) {
                Some(helper) => {
                    if helper.progress != line {
                        helper.progress = line;
                        helper.at = now;
                    }
                }
                None => {
                    helpers.push(Helper {
                        task_id,
                        label: view.label.clone(),
                        progress: line,
                        at: now,
                        started: now,
                    });
                    if helpers.len() > MAX_TRACKED_HELPERS {
                        helpers.remove(0);
                    }
                }
            }
            return;
        }
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
            StatusUpdate::ToolStart { kind, .. } => Some(kind.words().to_string()),
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
            StatusUpdate::Checklist { text } => {
                progress.plan = Some(text.clone());
                None
            }
            _ => None,
        };
        if let Some(step) = step.filter(|step| !step.trim().is_empty()) {
            progress.last_step = Some((step, now));
        }
    }

    /// The session's running work: `request` started `elapsed` ago (empty
    /// when no turn runs), plus what its status updates showed. Helpers
    /// started by an earlier turn are included; only the running turn's
    /// own are marked as its.
    pub fn snapshot(&self, session_id: &str, request: &str, elapsed: Duration) -> RunningWork {
        let mut work = RunningWork {
            request: request.to_string(),
            elapsed,
            ..RunningWork::default()
        };
        let turn = self.sessions.lock().ok().and_then(|sessions| {
            sessions.get(session_id).map(|progress| {
                work.last_step = progress
                    .last_step
                    .as_ref()
                    .map(|(step, at)| (step.clone(), at.elapsed()));
                work.plan = progress.plan.clone();
                progress.task_id
            })
        });
        if let Ok(helpers) = self.helpers.lock() {
            work.helpers = helpers
                .get(session_id)
                .into_iter()
                .flatten()
                .map(|helper| HelperProgress {
                    label: helper.label.clone(),
                    progress: helper.progress.clone(),
                    since: helper.at.elapsed(),
                    running_for: helper.started.elapsed(),
                    this_turn: turn == Some(helper.task_id),
                })
                .collect();
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

/// Whether `helper` (its card label, cut with "…" when long) is the
/// running item titled `title` (which names the helper in full).
fn helper_is(helper: &HelperProgress, title: &str) -> bool {
    let label = helper.label.trim();
    if label.is_empty() {
        return false;
    }
    match label.strip_suffix('…') {
        Some(stem) => title.contains(stem.trim_end()),
        None => title.ends_with(label),
    }
}

/// Where the running work stands, from its typed state: the request and how
/// long it has run, its latest step, then one line per helper and
/// background command with its current step and its time from the running
/// list, the same clock and numbers `/tasks` shows. A helper that is no
/// longer on that list keeps its last step, without a time.
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
    let mut listed = vec![false; work.helpers.len()];
    let mut items = Vec::new();
    for item in work.running.iter().filter(|item| item.work.is_some()) {
        let helper = (item.work == Some(ChatWorkKind::SubAgent))
            .then(|| {
                work.helpers
                    .iter()
                    .enumerate()
                    .find(|(i, helper)| !listed[*i] && helper_is(helper, &item.title))
            })
            .flatten();
        items.push(match helper {
            Some((i, helper)) => {
                listed[i] = true;
                format!(
                    "• {}: {} ({} so far)",
                    crate::presentation::word_preview(&helper.label, 60),
                    crate::presentation::word_preview(&helper.progress, 80),
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
    // A helper of this turn that is no longer listed keeps its last step;
    // one an earlier turn started and that is gone has finished.
    for (helper, _) in work
        .helpers
        .iter()
        .zip(&listed)
        .filter(|(helper, listed)| !**listed && helper.this_turn)
    {
        items.push(format!(
            "• {}: {}",
            crate::presentation::word_preview(&helper.label, 60),
            crate::presentation::word_preview(&helper.progress, 80),
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
                kind: aidaemon_core::card::ActivityKind::Run,
            },
        );
        live.record(
            "s",
            7,
            &StatusUpdate::Helper(aidaemon_core::card::HelperView::running(
                "Set up the blog",
                true,
                None,
            )),
        );
        live.record(
            "s",
            7,
            &StatusUpdate::Helper(aidaemon_core::card::HelperView::running(
                "Set up the blog",
                true,
                Some(aidaemon_core::card::Step::Doing(
                    aidaemon_core::card::ActivityKind::Write,
                )),
            )),
        );
        live.record(
            "s",
            7,
            &StatusUpdate::Checklist {
                text: "☑ one\n☐ two".into(),
            },
        );
        // A late update from an earlier turn changes nothing.
        live.record(
            "s",
            6,
            &StatusUpdate::ToolStart {
                name: "terminal".into(),
                summary: "old step".into(),
                kind: aidaemon_core::card::ActivityKind::Web,
            },
        );
        let work = live.snapshot("s", "Set up two demo sites", Duration::from_secs(300));
        assert_eq!(work.last_step.as_ref().unwrap().0, "Running a command");
        assert_eq!(work.plan.as_deref(), Some("☑ one\n☐ two"));
        assert_eq!(work.helpers.len(), 1);
        assert_eq!(work.helpers[0].progress, "Changing files");

        live.end("s", 6);
        assert!(live.snapshot("s", "x", Duration::ZERO).last_step.is_some());
        live.end("s", 7);
        assert!(live.snapshot("s", "x", Duration::ZERO).last_step.is_none());
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
            helpers: ["Slow count A", "Slow count B"]
                .into_iter()
                .map(|label| HelperProgress {
                    label: label.into(),
                    progress: "running a command".into(),
                    since: Duration::from_secs(4),
                    // Its own first-progress clock, which the reply no
                    // longer reads for a listed helper.
                    running_for: Duration::from_secs(10),
                    this_turn: true,
                })
                .collect(),
            running: list.items(now),
        };
        let reply = running_work_reply(&work);
        let tasks = list.lines(now);
        assert_eq!(tasks[0], "#1 Sub-agent: Slow count A (26s so far)");
        assert!(reply.starts_with("Still working on \"Count to 100 twice, slowly\" (55s so far)."));
        assert!(
            reply.contains("\n• Slow count A: running a command (26s so far)\n"),
            "{reply}"
        );
        assert!(
            reply.contains("\n• Slow count B: running a command (26s so far)"),
            "{reply}"
        );
        assert!(!reply.contains("10s"), "{reply}");
        assert!(reply.ends_with("I'll send the results when it's done."));

        // A helper no longer listed keeps its last step, without a time.
        let ended = RunningWork {
            running: Vec::new(),
            ..work
        };
        let reply = running_work_reply(&ended);
        assert!(
            reply.contains("• Slow count A: running a command\n"),
            "{reply}"
        );
    }
}
