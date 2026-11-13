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
    /// Why a run failed or stopped, from its typed records; `None` for an
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
    /// Its program ended with an error status, or was stopped (`None`).
    Exited { code: Option<i32> },
    /// It stopped because this machine ran out of something the owner can
    /// free, as the working agent declared in its typed verdict
    /// (`track_requirements`' `short_of`): its end offers the step that
    /// frees it ([`FreeStep`]).
    ShortOf(aidaemon_events::ShortResource),
}

/// The concrete next step offered for a resource a run ran out of: what
/// frees it, in plain words, and the command that would, when there is one
/// safe to name. Typed by the resource, never by an error's wording (live
/// 2026-11-13: DDEV stopped on "critically low Docker disk space" and the
/// answer only reported it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeStep {
    /// What ran out, as the owner reads it.
    pub what: &'static str,
    /// The step that frees it, offered.
    pub offer: &'static str,
    /// The command it runs, when the step is one command.
    pub command: Option<&'static str>,
}

impl FreeStep {
    pub fn of(resource: aidaemon_events::ShortResource) -> Self {
        use aidaemon_events::ShortResource;
        match resource {
            ShortResource::DockerDiskSpace => Self {
                what: "Docker ran out of disk space",
                offer: "I can free the space Docker isn't using (stopped containers, images no \
                        container uses, and its build cache) and then try again",
                command: Some("docker system prune"),
            },
            ShortResource::DiskSpace => Self {
                what: "this computer ran out of disk space",
                offer: "I can list what takes the most space in your home folder, so you can \
                        pick what to delete, and then try again",
                command: None,
            },
            ShortResource::Memory => Self {
                what: "this computer ran out of memory",
                offer: "I can list the programs using the most memory, so you can pick what to \
                        close, and then try again",
                command: None,
            },
        }
    }

    /// "Next step: … (`docker system prune`). Want me to?"
    pub fn line(&self) -> String {
        match self.command {
            Some(command) => format!("Next step: {} (`{command}`). Want me to?", self.offer),
            None => format!("Next step: {}. Want me to?", self.offer),
        }
    }
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
            Self::Exited { code: Some(code) } => {
                format!("Its program stopped with an error (exit status {code}).")
            }
            Self::Exited { code: None } => {
                "Its program was stopped before it finished.".to_string()
            }
            Self::ShortOf(resource) => {
                let step = FreeStep::of(*resource);
                format!("It stopped because {}. {}", step.what, step.line())
            }
        }
    }
}

impl RunEvidence {
    /// The one plain line under a finish header: why it failed, else the
    /// typed steps left unconfirmed. The unconfirmed ones belong to a run
    /// that finished (its own verdict was done and nothing contradicts it),
    /// so the line says only what is true of them, in words that agree with
    /// "finished": its report is the one record of them (live 2026-11-13:
    /// "I couldn't verify one of its steps" under a ✅ read as a failure).
    pub fn line(&self) -> Option<String> {
        if let Some(failure) = &self.failure {
            return Some(failure.reason());
        }
        match self.unconfirmed.as_slice() {
            [] => None,
            [one] => Some(format!("Its report is the only record of one step: {one}.")),
            many => Some(format!(
                "Its report is the only record of {} steps: {}.",
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
    /// When the owner's request that started it reached the runner, when a
    /// chat turn started it: a time "since your request" is measured from
    /// here.
    pub requested_at: Option<DateTime<Utc>>,
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

    /// Its one time set, once it ended ([`RunTimes`]).
    pub fn times(&self) -> Option<RunTimes> {
        self.ended_at
            .map(|ended| RunTimes::new(self.began(), ended, self.requested_at, self.waited))
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
            // The one measured line every surface shows for it
            // ([`RunTimes::line`]): the card, its result and the parent's
            // continuation all read the same labelled times. On the card
            // when a request started it, so the time from that request to
            // the run's start is said, not left as a gap between two
            // figures; a run no request started has it under its result.
            took: self
                .times()
                .filter(|times| times.requested.is_some())
                .map(|times| times.line()),
            since_request: self
                .requested_at
                .zip(self.ended_at)
                .map(|(requested, ended)| crate::spawn_tool::since_request_view(requested, ended)),
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
    groups: Mutex<HashMap<String, RunGroup>>,
}

/// The runs one turn started, in the order they started, and the ends of
/// those whose results are ready.
#[derive(Default)]
struct RunGroup {
    members: Vec<String>,
    ended: HashMap<String, GroupEnd>,
}

/// The group of the runs one chat turn started: every background run that
/// turn left running (helpers, commands, CLI agents) is answered together
/// once the last of them ends.
pub fn turn_group(chat: &str, task_id: &str) -> String {
    format!("{chat}#{task_id}")
}

/// One grouped run's end with its result, held until its group's last run
/// ends.
#[derive(Debug, Clone)]
pub struct GroupEnd {
    pub state: RunState,
    pub report: RunReport,
    /// Its own result message, rendered from its record: what the chat gets
    /// when the group's answer cannot be written.
    pub message: String,
}

/// What a run's end means for its group ([`RunStates::group_end`]).
#[derive(Debug)]
pub enum GroupTurn {
    /// It is not one of several runs a turn started: its result is its own
    /// answer.
    Alone,
    /// Other runs of its group still run: its result waits for theirs.
    Waiting,
    /// It was the group's last run: every member's end, in start order. Given
    /// once.
    Complete(Vec<GroupEnd>),
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
            requested_at: aidaemon_channels::turns::TurnOver::current()
                .map(|turn| turn.requested_at()),
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

    /// Run `id` is one of the runs of `group`: the runs one turn started.
    /// Their results are answered together once the last of them ends
    /// ([`Self::group_end`]).
    pub fn join_group(&self, id: &str, group: &str) {
        if !self.lock().contains_key(id) {
            return;
        }
        let mut groups = self
            .groups
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let members = &mut groups.entry(group.to_string()).or_default().members;
        if !members.iter().any(|member| member == id) {
            members.push(id.to_string());
        }
    }

    /// Run `id`'s result is ready (`end`). A run alone in its group, or in
    /// none, is [`GroupTurn::Alone`]; otherwise its end is held until every
    /// member's is, and the call that brings the last one gets them all, in
    /// start order, exactly once.
    pub fn group_end(&self, id: &str, end: GroupEnd) -> GroupTurn {
        let mut groups = self
            .groups
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(group) = groups
            .iter()
            .find(|(_, group)| group.members.iter().any(|member| member == id))
            .map(|(name, _)| name.clone())
        else {
            return GroupTurn::Alone;
        };
        let Some(entry) = groups.get_mut(&group) else {
            return GroupTurn::Alone;
        };
        if entry.members.len() < 2 {
            groups.remove(&group);
            return GroupTurn::Alone;
        }
        entry.ended.insert(id.to_string(), end);
        if entry
            .members
            .iter()
            .any(|member| !entry.ended.contains_key(member))
        {
            return GroupTurn::Waiting;
        }
        let Some(mut entry) = groups.remove(&group) else {
            return GroupTurn::Waiting;
        };
        GroupTurn::Complete(
            entry
                .members
                .iter()
                .filter_map(|member| entry.ended.remove(member))
                .collect(),
        )
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

    /// The runs drawn on card `card` (the card their starting turn kept),
    /// oldest first.
    pub fn on_card(&self, chat: &str, card: &str) -> Vec<RunState> {
        self.for_chat(chat)
            .into_iter()
            .filter(|state| matches!(&state.card, RunCard::Kept(kept) if kept == card))
            .collect()
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

/// The one set of times a run's record gives, and the only one any surface
/// shows for it: its work from start to end ("took", start-up left out:
/// its first step, else its start) and, when a request started it, that
/// request's time, so "started … after your request" and "… after your
/// request" are the same three instants seen two ways. All cut to the whole
/// second first, so the request's gap plus the run's own time is exactly
/// its time after the request (live 2026-11-13: "Done 43s after your
/// request" beside "Took 30s" left 13 s unexplained, and a group answer
/// quoted "took 3m 55s" while the card said "5m 54s after your request").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunTimes {
    /// Its work, start to end.
    pub run: MeasuredSpan,
    /// The owner's request that started it, never after its start.
    pub requested: Option<DateTime<Utc>>,
    /// How long it waited for the owner's answer on a card.
    pub waited: Duration,
}

impl RunTimes {
    pub fn new(
        began: DateTime<Utc>,
        ended: DateTime<Utc>,
        requested: Option<DateTime<Utc>>,
        waited: Duration,
    ) -> Self {
        use chrono::SubsecRound;
        let run = measured_span(began, ended);
        Self {
            run,
            requested: requested.map(|at| at.trunc_subsecs(0).min(run.from)),
            waited,
        }
    }

    /// Its work's own time: start to end.
    pub fn took(&self) -> chrono::Duration {
        self.run.took()
    }

    /// From the request to its start: the time before its work began.
    pub fn started_after_request(&self) -> Option<chrono::Duration> {
        self.requested.map(|requested| self.run.from - requested)
    }

    /// From the request to its end: what the owner means by "how long did
    /// it take".
    pub fn after_request(&self) -> Option<MeasuredSpan> {
        self.requested.map(|requested| MeasuredSpan {
            from: requested,
            to: self.run.to,
        })
    }

    /// The one labelled line for the owner and the parent: "Took 30s (8:17:58
    /// AM to 8:18:28 AM ET), started 13s after your request, so it ended 43s
    /// after your request." Each figure says what it measures; none is left
    /// for anyone to work out.
    pub fn line(&self) -> String {
        use crate::spawn_tool::{compact, owner_clock_range};
        let mut line = format!(
            "Took {} ({})",
            compact(self.took()),
            owner_clock_range(self.run.from, self.run.to)
        );
        if let (Some(gap), Some(whole)) = (self.started_after_request(), self.after_request()) {
            if gap.num_seconds() == 0 {
                line.push_str(", starting at your request");
            } else {
                line.push_str(&format!(
                    ", started {} after your request, so it ended {} after your request",
                    compact(gap),
                    compact(whole.took())
                ));
            }
        }
        line.push('.');
        if self.waited.as_secs() > 0 {
            line.push_str(&format!(
                " It waited {} of that for your answer.",
                aidaemon_tools::duration_format::compact_seconds(
                    i64::try_from(self.waited.as_secs()).unwrap_or(i64::MAX),
                    aidaemon_tools::duration_format::ZeroUnitStyle::Trim,
                )
            ));
        }
        line
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

    /// Live 2026-11-13: a background shell loop's card said "Done 43s
    /// after your request" and its result "Took 30s", leaving the 13 s
    /// before it started unexplained. The card's line is the record's one
    /// time set: how long it took, when after the request it started, and
    /// so when after the request it ended.
    #[test]
    fn the_cards_line_says_when_after_the_request_a_run_started() {
        use chrono::TimeZone;
        let requested = Utc.with_ymd_and_hms(2026, 11, 13, 13, 23, 36).unwrap()
            + chrono::Duration::milliseconds(748);
        let state = RunState {
            id: "loop".into(),
            chat: "chat".into(),
            label: "a shell loop".into(),
            phase: RunPhase::Succeeded,
            requested_at: Some(requested),
            started_at: requested + chrono::Duration::seconds(14),
            ended_at: Some(requested + chrono::Duration::milliseconds(42_729)),
            work_began: Some(requested + chrono::Duration::milliseconds(12_400)),
            waited: Duration::ZERO,
            step: None,
            evidence: RunEvidence::default(),
            card: RunCard::default(),
        };
        let times = state.times().unwrap();
        let line = state.view().took.expect("the card's measured line");
        assert_eq!(line, times.line());
        assert!(line.starts_with("Took 30s ("), "{line}");
        assert!(
            line.ends_with(", started 13s after your request, so it ended 43s after your request."),
            "{line}"
        );
        assert_eq!(
            times.started_after_request().unwrap() + times.took(),
            times.after_request().unwrap().took()
        );
        // No request on record: its time is under its result, not the card.
        let unrequested = RunState {
            requested_at: None,
            ..state
        };
        assert_eq!(unrequested.view().took, None);
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
