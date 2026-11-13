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

pub use aidaemon_core::card::RunPhase;
use aidaemon_core::card::{HelperView, Step};

/// The phase a child's typed end outcome gives.
pub fn phase_of(outcome: TaskOutcome) -> RunPhase {
    match outcome {
        TaskOutcome::Succeeded => RunPhase::Succeeded,
        TaskOutcome::Partial => RunPhase::Stopped,
        TaskOutcome::Failed => RunPhase::Failed,
    }
}

/// What the run's receipts left unproven: its own typed checklist steps
/// that declared a side effect and that no receipt confirmed. A step that
/// changes nothing needs none: the helper's report says what it found.
/// Only a run with more than one typed step lists them; a one-step run's
/// step is the run itself, which its phase already says.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunEvidence {
    pub unconfirmed: Vec<String>,
    /// Why a `Failed` run failed, from its typed records; `None` for an
    /// error, which says why itself.
    pub failure: Option<RunFailure>,
}

/// Why a helper's run failed, decided from its typed records (its checklist
/// steps and its receipts), never from its reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunFailure {
    /// It ended without changing anything, while each step of its own that
    /// needed a change was left undone: these steps (live 2026-11-13: a
    /// WordPress install helper looked at its folder, found it missing and
    /// stopped, reported "Blocked" in words, and its run read as finished).
    NothingDone { undone: Vec<String> },
}

impl RunFailure {
    /// The failure in one plain sentence.
    pub fn reason(&self) -> String {
        match self {
            Self::NothingDone { undone } if undone.is_empty() => {
                "It ended without doing any of its work: nothing was changed.".to_string()
            }
            Self::NothingDone { undone } => format!(
                "It ended without doing any of its work: nothing was changed, and {} undone: {}.",
                if undone.len() == 1 {
                    "this step is"
                } else {
                    "these steps are"
                },
                undone.join("; ")
            ),
        }
    }
}

impl RunEvidence {
    /// The one plain line under a finish header: why it failed, else the
    /// typed steps left unconfirmed.
    pub fn line(&self) -> Option<String> {
        if let Some(failure) = &self.failure {
            return Some(failure.reason());
        }
        match self.unconfirmed.as_slice() {
            [] => None,
            [one] => Some(format!("I couldn't verify one of its steps: {one}.")),
            many => Some(format!(
                "I couldn't verify {} of its steps: {}.",
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
    /// Its latest step while it runs.
    pub step: Option<Step>,
    pub evidence: RunEvidence,
    /// The chat card this run's line is drawn on after its turn: the card
    /// the turn that started it kept. A no-plan run's end edits exactly this
    /// card, never whichever card is newest.
    pub card: RunCard,
}

/// Which chat card a run's line is drawn on once the turn that started it
/// ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunCard {
    /// The starting turn has not kept its card yet; `before` is the chat's
    /// kept card when the run began (`None`: there was none), so the card
    /// is the first one the chat keeps after it.
    Pending { before: Option<String> },
    /// The message id of the card the starting turn kept.
    Kept(String),
}

impl Default for RunCard {
    fn default() -> Self {
        Self::Pending { before: None }
    }
}

impl RunState {
    /// When its work began: its first step, else its start.
    pub fn began(&self) -> DateTime<Utc> {
        self.work_began.unwrap_or(self.started_at)
    }

    /// When its work began and ended and how long it took, once it ended
    /// ([`measured_span`]).
    pub fn span(&self) -> Option<MeasuredSpan> {
        self.ended_at
            .map(|ended| measured_span(self.began(), ended))
    }

    /// How long it took, once it ended.
    pub fn took(&self) -> Option<chrono::Duration> {
        self.span().map(|span| span.took())
    }

    /// The helper as every card shows it.
    pub fn view(&self) -> HelperView {
        HelperView {
            label: self.label.clone(),
            phase: self.phase,
            background: true,
            step: match self.phase {
                RunPhase::Running => self.step.clone(),
                _ => None,
            },
            // Its measured time goes with its result, the parent's record
            // of it: the card says only how it ended.
            took: None,
        }
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
            card: RunCard::default(),
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
    pub fn step(&self, id: &str, step: Step) -> bool {
        let mut runs = self.lock();
        match runs.get_mut(id) {
            Some(entry) if entry.state.phase == RunPhase::Running => {
                entry.state.step = Some(step);
                true
            }
            _ => false,
        }
    }

    /// The chat's kept card when run `id` began: its own card is the first
    /// one the chat keeps after this. Ignored once the card is known.
    pub fn card_before(&self, id: &str, before: Option<String>) {
        if let Some(entry) = self.lock().get_mut(id) {
            if matches!(entry.state.card, RunCard::Pending { .. }) {
                entry.state.card = RunCard::Pending { before };
            }
        }
    }

    /// Run `id`'s line is drawn on card `message_id`, the card its starting
    /// turn kept. Set once: a later card belongs to a later turn.
    pub fn keep_card(&self, id: &str, message_id: &str) {
        if let Some(entry) = self.lock().get_mut(id) {
            if matches!(entry.state.card, RunCard::Pending { .. }) {
                entry.state.card = RunCard::Kept(message_id.to_string());
            }
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

/// A measured stretch of time as the owner reads it: both ends cut to the
/// whole second first, so the duration is always the difference of the
/// times shown (live 2026-11-13: a 20.4 s run read "20 seconds" beside
/// 2:21:16 to 2:21:37, which is 21).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasuredSpan {
    pub from: DateTime<Utc>,
    pub to: DateTime<Utc>,
}

impl MeasuredSpan {
    /// How long it took: `to - from`, never negative.
    pub fn took(&self) -> chrono::Duration {
        (self.to - self.from).max(chrono::Duration::zero())
    }
}

/// `from` to `to`, measured to the whole second ([`MeasuredSpan`]).
pub fn measured_span(from: DateTime<Utc>, to: DateTime<Utc>) -> MeasuredSpan {
    use chrono::SubsecRound;
    let from = from.trunc_subsecs(0);
    MeasuredSpan {
        from,
        to: to.trunc_subsecs(0).max(from),
    }
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
        assert!(runs.step("job-1", Step::Output("3".into())));
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
        assert!(!runs.step("job-1", Step::Output("4".into())));
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
        runs.step("job-7", Step::Output("9".into()));
        let listed = children.listed("chat-a");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].step, Some(Step::Output("9".into())));
        assert_eq!(listed[0].child.started_at, begun.started_at);
    }
}
