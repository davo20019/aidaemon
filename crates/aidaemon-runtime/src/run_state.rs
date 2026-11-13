//! One typed record per background helper run, owned by the runtime.
//!
//! A run is `Running` from its start until the runtime records how it ended
//! (`Succeeded`, `Failed` or `Stopped`), with the times the runtime
//! measured and the evidence its receipts left. Model text never sets any of
//! it. Every surface that shows a helper (the plan card's helper lines and
//! step marks, the finish header and its "Took" footer, `/tasks`) renders
//! from this record, and the result's delivery subscribes to the run's
//! terminal transition: it is pushed to the chat the moment the run ends,
//! never on the chat's next message.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use aidaemon_events::TaskOutcome;
use chrono::{DateTime, Utc};
use tokio::sync::mpsc;

/// Where a helper run stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPhase {
    Running,
    /// Its own end record says it did the work.
    Succeeded,
    /// It ended without doing the work: an error, a typed failure.
    Failed,
    /// It ended before the work was done: it stopped short by its own
    /// typed verdict, was cancelled, or made no progress for its limit.
    Stopped,
}

impl RunPhase {
    /// The phase a child's typed end outcome gives.
    pub fn ended(outcome: TaskOutcome) -> Self {
        match outcome {
            TaskOutcome::Succeeded => Self::Succeeded,
            TaskOutcome::Partial => Self::Stopped,
            TaskOutcome::Failed => Self::Failed,
        }
    }

    pub fn is_terminal(self) -> bool {
        self != Self::Running
    }

    /// The first line of the helper's result in the chat.
    pub fn headline(self) -> &'static str {
        match self {
            Self::Running => "\u{23f3} Running in the background",
            Self::Succeeded => "\u{2705} Finished in the background",
            Self::Stopped => "\u{26a0}\u{fe0f} Didn't finish in the background",
            Self::Failed => "\u{274c} Failed in the background",
        }
    }

    /// The helper's line on the plan card, after its name.
    pub fn card_word(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "done",
            Self::Stopped => "stopped before it finished",
            Self::Failed => "failed",
        }
    }
}

/// What the run's receipts left unproven: its own typed checklist steps no
/// receipt confirmed. Only a run with more than one typed step lists them;
/// a one-step run's step is the run itself, which its phase already says.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunEvidence {
    pub unconfirmed: Vec<String>,
}

impl RunEvidence {
    /// The one plain line under a finish header, when a typed step is
    /// unconfirmed.
    pub fn line(&self) -> Option<String> {
        match self.unconfirmed.as_slice() {
            [] => None,
            [one] => Some(format!("Its tools didn't confirm one step: {one}.")),
            many => Some(format!(
                "Its tools didn't confirm {} steps: {}.",
                many.len(),
                many.join("; ")
            )),
        }
    }
}

/// One helper run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunState {
    /// The background job id that names it everywhere.
    pub id: String,
    /// The chat it reports to.
    pub chat: String,
    /// The name the owner reads for it.
    pub label: String,
    pub phase: RunPhase,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    /// Its first tool step, when it took one: how long it took is measured
    /// from there, its start-up left out.
    pub work_began: Option<DateTime<Utc>>,
    /// How long it waited for the owner's answer on a card.
    pub waited: Duration,
    /// Its latest step, as its progress line says it.
    pub step: Option<String>,
    pub evidence: RunEvidence,
}

impl RunState {
    /// When its work began: its first step, else its start.
    pub fn began(&self) -> DateTime<Utc> {
        self.work_began.unwrap_or(self.started_at)
    }

    /// How long it took, once it ended.
    pub fn took(&self) -> Option<chrono::Duration> {
        self.ended_at
            .map(|ended| (ended - self.began()).max(chrono::Duration::zero()))
    }

    /// Its line on the plan card: its name, then its latest step while it
    /// runs, else how it ended.
    pub fn card_line(&self) -> (String, Option<String>) {
        let detail = match self.phase {
            RunPhase::Running => self.step.clone(),
            phase => Some(phase.card_word().to_string()),
        };
        (self.label.clone(), detail)
    }
}

/// What the run's end carries to its delivery: the helper's own report (its
/// text; never what decides its phase) and whether that text is trusted.
#[derive(Debug, Clone)]
pub struct RunReport {
    pub text: String,
    pub untrusted: bool,
}

/// A run reached a terminal phase.
#[derive(Debug, Clone)]
pub struct RunEnded {
    pub state: RunState,
    pub report: RunReport,
}

/// Every helper run of one runtime. A run stays listed after it ends (until
/// [`RunStates::forget`]), so a surface drawn late still reads how it ended.
#[derive(Default)]
pub struct RunStates {
    runs: Mutex<HashMap<String, Entry>>,
    order: std::sync::atomic::AtomicU64,
}

struct Entry {
    seq: u64,
    state: RunState,
    on_end: Option<mpsc::UnboundedSender<RunEnded>>,
}

impl RunStates {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Entry>> {
        self.runs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Start run `id` in `chat`. `on_end` subscribes to its terminal
    /// transition: it receives the run once, the moment it ends.
    pub fn begin(
        &self,
        id: &str,
        chat: &str,
        label: &str,
        on_end: Option<mpsc::UnboundedSender<RunEnded>>,
    ) -> RunState {
        let state = RunState {
            id: id.to_string(),
            chat: chat.to_string(),
            label: label.to_string(),
            phase: RunPhase::Running,
            started_at: Utc::now(),
            ended_at: None,
            work_began: None,
            waited: Duration::ZERO,
            step: None,
            evidence: RunEvidence::default(),
        };
        let seq = self
            .order
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.lock().insert(
            id.to_string(),
            Entry {
                seq,
                state: state.clone(),
                on_end,
            },
        );
        state
    }

    /// The running run's latest step. Ignored once it ended.
    pub fn step(&self, id: &str, step: &str) -> bool {
        let mut runs = self.lock();
        match runs.get_mut(id) {
            Some(entry) if entry.state.phase == RunPhase::Running => {
                entry.state.step = Some(step.to_string());
                true
            }
            _ => false,
        }
    }

    /// Record how run `id` ended, with what the runtime measured, and push
    /// it to its subscriber. A run ends once: a second end is ignored.
    pub fn finish(
        &self,
        id: &str,
        phase: RunPhase,
        measured: RunMeasure,
        evidence: RunEvidence,
        report: RunReport,
    ) -> Option<RunState> {
        debug_assert!(phase.is_terminal());
        let (state, on_end) = {
            let mut runs = self.lock();
            let entry = runs.get_mut(id)?;
            if entry.state.phase.is_terminal() {
                return None;
            }
            entry.state.phase = phase;
            entry.state.ended_at = Some(measured.ended_at);
            entry.state.work_began = measured.work_began;
            entry.state.waited = measured.waited;
            entry.state.evidence = evidence;
            (entry.state.clone(), entry.on_end.take())
        };
        if let Some(on_end) = on_end {
            let _ = on_end.send(RunEnded {
                state: state.clone(),
                report,
            });
        }
        Some(state)
    }

    pub fn get(&self, id: &str) -> Option<RunState> {
        self.lock().get(id).map(|entry| entry.state.clone())
    }

    /// The chat's runs, oldest first.
    pub fn for_chat(&self, chat: &str) -> Vec<RunState> {
        let runs = self.lock();
        let mut listed: Vec<(u64, RunState)> = runs
            .values()
            .filter(|entry| entry.state.chat == chat)
            .map(|entry| (entry.seq, entry.state.clone()))
            .collect();
        listed.sort_by_key(|(seq, _)| *seq);
        listed.into_iter().map(|(_, state)| state).collect()
    }

    /// The chat's runs still running, oldest first.
    pub fn running(&self, chat: &str) -> Vec<RunState> {
        self.for_chat(chat)
            .into_iter()
            .filter(|state| state.phase == RunPhase::Running)
            .collect()
    }

    /// Drop an ended run's record.
    pub fn forget(&self, id: &str) {
        let mut runs = self.lock();
        if runs
            .get(id)
            .is_some_and(|entry| entry.state.phase.is_terminal())
        {
            runs.remove(id);
        }
    }
}

tokio::task_local! {
    static CURRENT_RUN: std::cell::RefCell<Option<String>>;
}

/// Run `fut` as background run `id`: the child it starts first takes the
/// id ([`take_current_run`]), so `/tasks` lists that child from the run's
/// record.
pub async fn within_run<F: std::future::Future>(id: &str, fut: F) -> F::Output {
    CURRENT_RUN
        .scope(std::cell::RefCell::new(Some(id.to_string())), fut)
        .await
}

/// The run id [`within_run`] set, taken once: the helper itself gets it,
/// never a child it starts in turn.
pub fn take_current_run() -> Option<String> {
    CURRENT_RUN
        .try_with(|run| run.borrow_mut().take())
        .ok()
        .flatten()
}

/// What the runtime measured of a run when it ended.
#[derive(Debug, Clone, Copy)]
pub struct RunMeasure {
    pub ended_at: DateTime<Utc>,
    pub work_began: Option<DateTime<Utc>>,
    pub waited: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> RunReport {
        RunReport {
            text: "Done.".to_string(),
            untrusted: false,
        }
    }

    fn measure() -> RunMeasure {
        RunMeasure {
            ended_at: Utc::now(),
            work_began: None,
            waited: Duration::ZERO,
        }
    }

    /// A run's end is pushed to its subscriber once, at the transition, and
    /// later steps or ends change nothing.
    #[tokio::test]
    async fn a_runs_end_is_pushed_once_and_is_final() {
        let runs = RunStates::default();
        let (tx, mut rx) = mpsc::unbounded_channel();
        runs.begin("job-1", "chat-a", "Slow count", Some(tx));
        assert!(runs.step("job-1", "running a command: 3"));
        assert!(rx.try_recv().is_err(), "nothing is pushed while it runs");
        let ended = runs
            .finish(
                "job-1",
                RunPhase::Succeeded,
                measure(),
                RunEvidence::default(),
                report(),
            )
            .expect("it ends");
        assert_eq!(ended.phase, RunPhase::Succeeded);
        let pushed = rx.try_recv().expect("pushed at the transition");
        assert_eq!(pushed.state, ended);
        assert!(runs
            .finish(
                "job-1",
                RunPhase::Failed,
                measure(),
                RunEvidence::default(),
                report()
            )
            .is_none());
        assert!(!runs.step("job-1", "late step"));
        assert_eq!(runs.get("job-1").unwrap().phase, RunPhase::Succeeded);
        assert!(runs.running("chat-a").is_empty());
    }

    /// `/tasks` lists a background helper from its run record: the step
    /// and start the card shows, not a copy of its own.
    #[test]
    fn tasks_list_a_helper_from_its_run_record() {
        let children = std::sync::Arc::new(crate::spawn::RunningChildren::default());
        let runs = children.runs();
        let begun = runs.begin("job-7", "chat-a", "Slow count", None);
        let _entry = children.enter("chat-a", "Slow count", "Count", "child-7", Some("job-7"));
        runs.step("job-7", "running a command: 9");
        let listed = children.listed("chat-a");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].step.as_deref(), Some("running a command: 9"));
        assert_eq!(listed[0].child.started_at, begun.started_at);
    }
}
