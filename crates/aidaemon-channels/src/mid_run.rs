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

use crate::ports::{HelperProgress, RunningWork};

/// How many helpers a reply lists.
const MAX_HELPERS: usize = 6;

#[derive(Debug, Default)]
struct Progress {
    task_id: u64,
    last_step: Option<(String, Instant)>,
    plan: Option<String>,
    /// Each helper: its label, latest progress, when that came, and when
    /// it was first seen.
    helpers: Vec<Helper>,
}

#[derive(Debug)]
struct Helper {
    label: String,
    progress: String,
    at: Instant,
    started: Instant,
}

/// Each session's running work as its status updates showed it.
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
        if let Ok(mut sessions) = self.sessions.lock() {
            if sessions
                .get(session_id)
                .is_some_and(|p| p.task_id == task_id)
            {
                sessions.remove(session_id);
            }
        }
    }

    /// Note what `update` says about `task_id`'s work.
    pub fn record(&self, session_id: &str, task_id: u64, update: &StatusUpdate) {
        let Ok(mut sessions) = self.sessions.lock() else {
            return;
        };
        let Some(progress) = sessions
            .get_mut(session_id)
            .filter(|progress| progress.task_id == task_id)
        else {
            return;
        };
        let now = Instant::now();
        let step = match update {
            StatusUpdate::ToolStart { name, summary } => Some(if summary.trim().is_empty() {
                name.clone()
            } else {
                summary.clone()
            }),
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
            StatusUpdate::SubAgentProgress {
                label,
                progress: line,
            } => {
                match progress.helpers.iter_mut().find(|h| h.label == *label) {
                    Some(helper) => {
                        helper.progress = line.clone();
                        helper.at = now;
                    }
                    None => progress.helpers.push(Helper {
                        label: label.clone(),
                        progress: line.clone(),
                        at: now,
                        started: now,
                    }),
                }
                None
            }
            _ => None,
        };
        if let Some(step) = step.filter(|step| !step.trim().is_empty()) {
            progress.last_step = Some((step, now));
        }
    }

    /// The session's running work: `request` started `elapsed` ago, plus
    /// what its status updates showed.
    pub fn snapshot(&self, session_id: &str, request: &str, elapsed: Duration) -> RunningWork {
        let mut work = RunningWork {
            request: request.to_string(),
            elapsed,
            ..RunningWork::default()
        };
        let Ok(sessions) = self.sessions.lock() else {
            return work;
        };
        if let Some(progress) = sessions.get(session_id) {
            work.last_step = progress
                .last_step
                .as_ref()
                .map(|(step, at)| (step.clone(), at.elapsed()));
            work.plan = progress.plan.clone();
            work.helpers = progress
                .helpers
                .iter()
                .map(|helper| HelperProgress {
                    label: helper.label.clone(),
                    progress: helper.progress.clone(),
                    since: helper.at.elapsed(),
                    running_for: helper.started.elapsed(),
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

/// Where the running work stands, from its typed state: the request and how
/// long it has run, its latest step, and each helper's job, progress and
/// running time.
pub fn running_work_status(work: &RunningWork) -> String {
    let mut lines = vec![format!(
        "Still working on \"{}\" ({} so far).",
        crate::presentation::word_preview(&work.request, 80),
        ago(work.elapsed)
    )];
    if let Some((step, since)) = &work.last_step {
        lines.push(format!(
            "Latest step: {} ({} ago).",
            crate::presentation::word_preview(step, 120),
            ago(*since)
        ));
    }
    for helper in work.helpers.iter().take(MAX_HELPERS) {
        lines.push(format!(
            "• {}: {} (running {})",
            crate::presentation::word_preview(&helper.label, 60),
            crate::presentation::word_preview(&helper.progress, 80),
            ago(helper.running_for)
        ));
    }
    lines.join("\n")
}

/// The reply to a question about the running work when the agent gave no
/// answer: what the state itself says, in plain words.
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
            },
        );
        live.record(
            "s",
            7,
            &StatusUpdate::SubAgentProgress {
                label: "Set up the blog".into(),
                progress: "step 2 of 4".into(),
            },
        );
        live.record(
            "s",
            7,
            &StatusUpdate::SubAgentProgress {
                label: "Set up the blog".into(),
                progress: "step 3 of 4".into(),
            },
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
            },
        );
        let work = live.snapshot("s", "Set up two demo sites", Duration::from_secs(300));
        assert_eq!(work.last_step.as_ref().unwrap().0, "installing the site");
        assert_eq!(work.plan.as_deref(), Some("☑ one\n☐ two"));
        assert_eq!(work.helpers.len(), 1);
        assert_eq!(work.helpers[0].progress, "step 3 of 4");

        live.end("s", 6);
        assert!(live.snapshot("s", "x", Duration::ZERO).last_step.is_some());
        live.end("s", 7);
        assert!(live.snapshot("s", "x", Duration::ZERO).last_step.is_none());
    }

    #[test]
    fn the_fallback_reply_says_the_state_in_plain_words() {
        let work = RunningWork {
            request: "Set up two demo sites".into(),
            elapsed: Duration::from_secs(330),
            last_step: Some(("installing the blog".into(), Duration::from_secs(20))),
            plan: None,
            helpers: vec![HelperProgress {
                label: "Set up the blog".into(),
                progress: "step 2 of 4".into(),
                since: Duration::from_secs(5),
                running_for: Duration::from_secs(130),
            }],
            ..RunningWork::default()
        };
        let reply = running_work_reply(&work);
        assert!(reply.starts_with("Still working on \"Set up two demo sites\" (5m 30s so far)."));
        assert!(reply.contains("Latest step: installing the blog (20s ago)."));
        assert!(reply.contains("• Set up the blog: step 2 of 4 (running 2m 10s)"));
        assert!(reply.ends_with("I'll send the results when it's done."));
        assert!(!reply.contains('#'));
    }
}
