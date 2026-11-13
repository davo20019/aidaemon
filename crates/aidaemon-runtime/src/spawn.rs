//! Spawning sub-agents: the running-children registry `/status` and `/tasks`
//! read, how a child's progress and end reach its parent, what a stopped
//! child had done, and the plain child's prompt. Building and running the
//! child itself is the host's.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};
use std::time::Duration;

use aidaemon_agents::specialist::SpecialistKind;
use aidaemon_channels::context::ChannelContext;
use aidaemon_channels::ports::{AgentResponseEnvelope, OutboundRouter};
use aidaemon_events::traits::AgentRole;
use aidaemon_events::{EventType, TaskEndData, TaskOutcome};
use aidaemon_tools::semantics::UserRole;
use aidaemon_tools::status::StatusUpdate;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tracing::warn;

/// What a stopped child had already done, from its own tool records.
#[derive(Debug, Default)]
pub struct ChildWorkReport {
    /// Calls that finished and may have changed something.
    pub changed: Vec<String>,
    /// Calls still running when the child was stopped.
    pub unfinished: Vec<String>,
    /// The records could not be read in full (unreadable, or more than the
    /// report reads), so the lists may be missing actions.
    pub unknown: bool,
}

impl ChildWorkReport {
    /// Whether the child may have changed something before it stopped.
    pub fn may_have_changed_something(&self) -> bool {
        self.unknown || !self.changed.is_empty() || !self.unfinished.is_empty()
    }
}

/// Whether a child's run saw no untrusted content: its in-memory flag is
/// unset and its record of tainted turns (its session is fresh per spawn) is
/// readable and empty. An unreadable record counts as tainted, and so does a
/// taint the child saw but could not record.
pub fn child_run_is_clean(read_untrusted: bool, record: &anyhow::Result<Vec<String>>) -> bool {
    !read_untrusted && matches!(record, Ok(turns) if turns.is_empty())
}

/// Who authored a child's result: `Untrusted` unless its run was clean.
pub fn child_result_trust(
    read_untrusted: bool,
    record: &anyhow::Result<Vec<String>>,
) -> aidaemon_core::SourceTrust {
    if child_run_is_clean(read_untrusted, record) {
        aidaemon_core::SourceTrust::Trusted
    } else {
        aidaemon_core::SourceTrust::Untrusted
    }
}

/// How much of a mission the roster shows.
pub const ROSTER_MISSION_CHARS: usize = 120;

#[derive(Debug)]
pub struct SpawnChildResult {
    /// The child's own session.
    pub session: String,
    pub response: String,
    pub outcome: TaskOutcome,
    /// `Untrusted` when the child's run read (or started with) untrusted
    /// content: its result carries that taint back to the parent.
    pub trust: aidaemon_core::SourceTrust,
    /// How its run ended, as the run-state record holds it ([`run_phase`]).
    pub phase: crate::run_state::RunPhase,
    /// Its typed steps no receipt confirmed.
    pub evidence: crate::run_state::RunEvidence,
}

/// How a child's run ended, the one place that decides it, from typed facts
/// only: its end record's outcome, and its run aggregate. A run whose
/// outcome is partial only for want of receipts (it ended normally with its
/// own typed verdict `done`, nothing it started still runs, the owner
/// declined nothing, and it set none of its steps aside) succeeded; the
/// typed steps that declared a side effect and that no receipt matched are
/// its evidence, listed only when it declared more than one step (a
/// one-step run's step is the run itself).
pub fn run_phase(
    outcome: TaskOutcome,
    end: Option<&TaskEndData>,
    aggregate: Option<&aidaemon_events::RunAggregate>,
) -> (crate::run_state::RunPhase, crate::run_state::RunEvidence) {
    use crate::run_state::{RunEvidence, RunPhase};
    use aidaemon_events::RunObligationState;
    let (Some(end), Some(aggregate)) = (end, aggregate) else {
        return (crate::run_state::phase_of(outcome), RunEvidence::default());
    };
    let checklist: Vec<&aidaemon_events::RunObligation> = aggregate
        .obligations
        .iter()
        .filter(|(id, _)| id.contains("/obligation:checklist:"))
        .map(|(_, obligation)| obligation)
        .collect();
    if let Some(failure) = nothing_done(outcome, &checklist, aggregate) {
        return (
            RunPhase::Failed,
            RunEvidence {
                unconfirmed: Vec::new(),
                failure: Some(failure),
            },
        );
    }
    let only_receipts_missing = outcome == TaskOutcome::Partial
        && end.status == aidaemon_events::TaskStatus::Completed
        && aggregate.executor_claimed_outcome
            == Some(aidaemon_events::ExecutorClaimedOutcome::Done)
        && !aggregate.has_background_in_flight()
        && !aggregate.owner_declined()
        && checklist.iter().all(|obligation| {
            matches!(
                obligation.state,
                RunObligationState::Satisfied | RunObligationState::Pending
            )
        });
    if !only_receipts_missing {
        return (crate::run_state::phase_of(outcome), RunEvidence::default());
    }
    let unconfirmed = if checklist.len() > 1 {
        let mut pending: Vec<(&String, String)> = aggregate
            .obligations
            .iter()
            .filter(|(id, obligation)| {
                // A step that changes nothing needs no receipt: the
                // helper's report says what it found.
                id.contains("/obligation:checklist:")
                    && obligation.state == RunObligationState::Pending
                    && !obligation.required_effect.is_empty()
            })
            .map(|(id, obligation)| {
                // The step in its own words: never the internal demand text
                // (live 2026-11-13: "checklist item 2 … needs a succeeded
                // receipt carrying local_workspace_write|process_state").
                (
                    id,
                    obligation
                        .step
                        .as_deref()
                        .map(plain_step)
                        .unwrap_or_else(|| "a step it listed".to_string()),
                )
            })
            .collect();
        pending.sort_by(|a, b| a.0.cmp(b.0));
        pending.into_iter().map(|(_, summary)| summary).collect()
    } else {
        Vec::new()
    };
    (
        RunPhase::Succeeded,
        RunEvidence {
            unconfirmed,
            failure: None,
        },
    )
}

/// A step's own words as the owner reads them: one line, without a
/// trailing period.
fn plain_step(step: &str) -> String {
    let line = step.split_whitespace().collect::<Vec<_>>().join(" ");
    line.trim_end_matches('.').to_string()
}

/// A run that did none of its work, from its typed records: it did not
/// succeed, it declared steps that need a change, none of them was done, no
/// call of its changed anything, and neither the owner's answer nor work
/// still running explains the stop. Such a run failed: whatever its reply
/// says, nothing it was sent to do happened.
fn nothing_done(
    outcome: TaskOutcome,
    checklist: &[&aidaemon_events::RunObligation],
    aggregate: &aidaemon_events::RunAggregate,
) -> Option<crate::run_state::RunFailure> {
    use aidaemon_events::RunObligationState;
    if outcome == TaskOutcome::Succeeded
        || aggregate.owner_declined()
        || aggregate.has_background_in_flight()
    {
        return None;
    }
    let effect_steps: Vec<&&aidaemon_events::RunObligation> = checklist
        .iter()
        .filter(|obligation| !obligation.required_effect.is_empty())
        .collect();
    let a_step_done = effect_steps
        .iter()
        .any(|obligation| obligation.state == RunObligationState::Satisfied);
    let changed_something = aggregate.operations.values().any(|operation| {
        operation.outcome == Some(aidaemon_tools::semantics::ToolOutcomeStatus::Succeeded)
            && !operation.mutation_effects.is_empty()
    });
    if effect_steps.is_empty() || a_step_done || changed_something {
        return None;
    }
    let mut undone: Vec<(&String, String)> = effect_steps
        .iter()
        .map(|obligation| {
            (
                &obligation.id,
                obligation
                    .step
                    .as_deref()
                    .map(plain_step)
                    .unwrap_or_else(|| "a step it listed".to_string()),
            )
        })
        .collect();
    undone.sort_by(|a, b| a.0.cmp(b.0));
    Some(crate::run_state::RunFailure::NothingDone {
        undone: undone.into_iter().map(|(_, summary)| summary).collect(),
    })
}

#[derive(Debug)]
pub struct SalvagedTaskOutcome {
    pub status: String,
    pub details: String,
}

/// The child's own latest end record, however many records follow it,
/// retrying briefly on SQLite contention. An error means the record could
/// not be read, which is not the same as the child having succeeded.
pub async fn latest_child_task_end(
    event_store: &aidaemon_events::EventStore,
    child_session: &str,
) -> anyhow::Result<Option<TaskEndData>> {
    const MAX_BUSY_RETRIES: usize = 3;
    let mut busy_retries = 0_usize;
    let events = loop {
        match event_store
            .query_recent_task_ends(child_session, false, 1)
            .await
        {
            Ok(events) => break events,
            Err(error) if busy_retries < MAX_BUSY_RETRIES && is_sqlite_busy(&error) => {
                tokio::time::sleep(Duration::from_millis(25_u64 << busy_retries)).await;
                busy_retries += 1;
            }
            Err(error) => return Err(error),
        }
    };
    // A record that is there but cannot be parsed is unreadable, never a
    // missing (and so never a successful) end.
    events
        .first()
        .map(|event| event.parse_data::<TaskEndData>())
        .transpose()
}

/// A child's outcome comes from its own end record. A reply with no readable
/// record is unverified, so it is never reported as a success.
pub fn resolve_child_outcome(
    replied: bool,
    record: &anyhow::Result<Option<TaskEndData>>,
) -> TaskOutcome {
    match record {
        Ok(Some(data)) => data.effective_outcome(),
        _ if replied => TaskOutcome::Partial,
        _ => TaskOutcome::Failed,
    }
}

/// The sub-agents running now, each under the chat it reports to (its
/// approval route): `/status` and `/tasks` list them by their title.
#[derive(Default)]
pub struct RunningChildren {
    next: std::sync::atomic::AtomicU64,
    entries: std::sync::Mutex<Vec<ListedChild>>,
    /// Each chat's running turn (its task id), so a child is listed under
    /// the request that started it.
    requests: std::sync::Mutex<std::collections::HashMap<String, String>>,
    /// The tree's background helper runs.
    runs: Arc<crate::run_state::RunStates>,
}

/// One running child as listed: its chat, its title and the task it was
/// given (which tells siblings with the same title apart).
pub struct ListedChild {
    id: u64,
    session_id: String,
    title: String,
    task: String,
    started_at: chrono::DateTime<chrono::Utc>,
    /// The child's own session, where its background commands run.
    child_session: String,
    /// The chat turn that was running when it started.
    request: Option<String>,
    /// Stops the child when the owner asks.
    stop: tokio_util::sync::CancellationToken,
    /// Its latest step, as its progress line says it.
    step: ChildStep,
    /// The background run it is, when it is one: its record gives its
    /// step and start.
    run: Option<String>,
}

/// A running child's latest step, written by its status channel.
pub type ChildStep = Arc<std::sync::Mutex<Option<aidaemon_core::card::Step>>>;

/// A listed child, as the chat's running work shows it.
pub struct ListedChildView {
    pub key: String,
    pub child: aidaemon_channels::ports::RunningChild,
    pub request: Option<String>,
    /// Its latest step, as `/tasks` and the status reply give it.
    pub step: Option<aidaemon_core::card::Step>,
}

/// A chat's running turn, recorded while it runs.
pub struct RunningRequest {
    registry: Arc<RunningChildren>,
    session_id: String,
    task_id: String,
}

impl Drop for RunningRequest {
    fn drop(&mut self) {
        let mut requests = self
            .registry
            .requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if requests.get(&self.session_id) == Some(&self.task_id) {
            requests.remove(&self.session_id);
        }
    }
}

/// One running child's place in [`RunningChildren`], removed when dropped
/// however the child ends.
pub struct RunningChildEntry {
    registry: Arc<RunningChildren>,
    id: u64,
    stop: tokio_util::sync::CancellationToken,
    step: ChildStep,
}

impl RunningChildEntry {
    /// Cancelled when the owner stops this child.
    pub fn stopped(&self) -> tokio_util::sync::CancellationToken {
        self.stop.clone()
    }

    /// Where the child's status channel writes its latest step.
    pub fn step(&self) -> ChildStep {
        self.step.clone()
    }
}

impl RunningChildren {
    /// The tree's background helper runs.
    pub fn runs(&self) -> Arc<crate::run_state::RunStates> {
        self.runs.clone()
    }

    pub fn enter(
        self: &Arc<Self>,
        session_id: &str,
        mission: &str,
        task: &str,
        child_session: &str,
        run: Option<&str>,
    ) -> RunningChildEntry {
        let id = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let request = self
            .requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(session_id)
            .cloned();
        let stop = tokio_util::sync::CancellationToken::new();
        let step = ChildStep::default();
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(ListedChild {
                id,
                session_id: session_id.to_string(),
                title: child_status_label(mission),
                task: task.to_string(),
                started_at: chrono::Utc::now(),
                child_session: child_session.to_string(),
                request,
                stop: stop.clone(),
                step: step.clone(),
                run: run.map(str::to_string),
            });
        RunningChildEntry {
            registry: self.clone(),
            id,
            stop,
            step,
        }
    }

    /// Record `task_id` as the chat's running turn until the guard drops.
    pub fn begin_request(self: &Arc<Self>, session_id: &str, task_id: &str) -> RunningRequest {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(session_id.to_string(), task_id.to_string());
        RunningRequest {
            registry: self.clone(),
            session_id: session_id.to_string(),
            task_id: task_id.to_string(),
        }
    }

    /// The chat's running turn, if one is recorded.
    pub fn current_request(&self, session_id: &str) -> Option<String> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(session_id)
            .cloned()
    }

    /// Ask the listed child `key` to stop; its child session when it was
    /// listed.
    pub fn stop(&self, key: &str) -> Option<String> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let child = entries.iter().find(|child| child_key(child.id) == key)?;
        child.stop.cancel();
        Some(child.child_session.clone())
    }

    /// Whether the child `key` is still listed (running).
    pub fn is_listed(&self, key: &str) -> bool {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .any(|child| child_key(child.id) == key)
    }

    /// The chat's running children. Children sharing a title (two parallel
    /// helpers given the same mission line) are told apart by the part of
    /// their task that differs.
    pub fn for_session(&self, session_id: &str) -> Vec<aidaemon_channels::ports::RunningChild> {
        self.listed(session_id)
            .into_iter()
            .map(|view| view.child)
            .collect()
    }

    /// [`Self::for_session`] with each child's key, own session and request.
    pub fn listed(&self, session_id: &str) -> Vec<ListedChildView> {
        let entries = self
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let listed: Vec<&ListedChild> = entries
            .iter()
            .filter(|child| child.session_id == session_id)
            .collect();
        listed
            .iter()
            .map(|child| {
                let siblings: Vec<&str> = listed
                    .iter()
                    .filter(|other| other.id != child.id && other.title == child.title)
                    .map(|other| other.task.as_str())
                    .collect();
                let label = match distinguishing_detail(&child.task, &siblings) {
                    Some(detail) => format!("{} ({detail})", child.title),
                    None => child.title.clone(),
                };
                // A background helper is shown from its run record.
                let run = child.run.as_deref().and_then(|id| self.runs.get(id));
                let step = match run.as_ref() {
                    Some(run) => run.step.clone(),
                    None => child
                        .step
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone(),
                };
                ListedChildView {
                    key: child_key(child.id),
                    child: aidaemon_channels::ports::RunningChild {
                        label,
                        started_at: run.map_or(child.started_at, |run| run.started_at),
                    },
                    request: child.request.clone(),
                    step,
                }
            })
            .collect()
    }
}

pub fn child_key(id: u64) -> String {
    format!("agent:{id}")
}

/// The words of `task` that tell it apart from `siblings`, starting one
/// word before the first difference; `None` with no siblings or no
/// difference.
pub fn distinguishing_detail(task: &str, siblings: &[&str]) -> Option<String> {
    const WORDS: usize = 5;
    const MAX_CHARS: usize = 50;
    if siblings.is_empty() {
        return None;
    }
    let words: Vec<&str> = task.split_whitespace().collect();
    let shared = siblings
        .iter()
        .map(|other| {
            words
                .iter()
                .zip(other.split_whitespace())
                .take_while(|(a, b)| **a == *b)
                .count()
        })
        .max()
        .unwrap_or(0);
    if shared >= words.len() {
        return None;
    }
    let start = shared.saturating_sub(1);
    let end = (start + WORDS).min(words.len());
    let mut detail = words[start..end].join(" ");
    if detail.chars().count() > MAX_CHARS {
        detail = detail.chars().take(MAX_CHARS - 1).collect::<String>();
        detail = format!("{}…", detail.trim_end());
    } else if end < words.len() {
        detail.push('…');
    }
    if start > 0 {
        detail.insert(0, '…');
    }
    Some(detail)
}

impl Drop for RunningChildEntry {
    fn drop(&mut self) {
        self.registry
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .retain(|child| child.id != self.id);
    }
}

/// A child that ended with a refused write it never made did not do that
/// work, whatever its own run concluded.
pub fn settle_coordinated_outcome(outcome: TaskOutcome, unlanded_write: bool) -> TaskOutcome {
    if outcome == TaskOutcome::Succeeded && unlanded_write {
        TaskOutcome::Partial
    } else {
        outcome
    }
}

/// How often a running child checks how long it has been idle.
pub const CHILD_LIMIT_CHECK: Duration = Duration::from_millis(250);
/// How many entries a stopped child's report lists per section.
pub const MAX_REPORTED_ACTIONS: usize = 20;
pub const REPORTED_ACTION_CHARS: usize = 160;
/// How many tool records (calls and results) a stopped child's report reads.
pub const MAX_REPORT_TOOL_RECORDS: usize = 5_000;

/// The status channel a child reports through. The parent's live card shows
/// the parent's own checklist: a child's checklist is the child's plan, so it
/// never reaches the parent's card (two parallel children used to overwrite
/// it in turn, leaving whichever wrote last looking like the whole task).
/// Each of the child's steps (a tool starting, a step handed on, the line its
/// command printed) becomes that child's [`HelperView`], so the
/// parent's card and its status reply say what each helper is doing as a
/// typed kind. A tool ending is not a step. Every other update passes
/// through unchanged.
///
/// [`HelperView`]: aidaemon_core::card::HelperView
pub fn child_status_sender(
    parent: Option<mpsc::Sender<StatusUpdate>>,
    mission: &str,
    background: bool,
    step: Option<ChildStep>,
) -> Option<mpsc::Sender<StatusUpdate>> {
    use aidaemon_core::card::{HelperView, Step};
    let parent = parent?;
    let label = child_status_label(mission);
    let (tx, mut rx) = mpsc::channel::<StatusUpdate>(64);
    tokio::spawn(async move {
        while let Some(update) = rx.recv().await {
            let own_step = match &update {
                // Between its steps the helper keeps its latest one: its
                // thinking is no step of its own.
                StatusUpdate::Checklist(_)
                | StatusUpdate::ToolComplete { .. }
                | StatusUpdate::Thinking(_) => continue,
                StatusUpdate::ToolStart { kind, .. } => Some(Step::Doing(*kind)),
                // A child's step that went on in the background is that
                // child's progress, never the parent's own step (live
                // 2026-11-13: "… goes on in the background" stood on the
                // card as if the turn itself had handed off).
                StatusUpdate::StepHandedOff { kind } => Some(Step::HandedOff(*kind)),
                // What the child's running command printed last is that
                // child's progress line, as it runs.
                StatusUpdate::CommandOutput { line, .. } => Some(Step::Output(line.clone())),
                _ => None,
            };
            let update = match own_step {
                Some(own_step) => {
                    // Its own latest step, for `/tasks`.
                    if let Some(slot) = step.as_ref() {
                        *slot
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            Some(own_step.clone());
                    }
                    StatusUpdate::Helper(HelperView::running(&label, background, Some(own_step)))
                }
                None => update,
            };
            if parent.send(update).await.is_err() {
                break;
            }
        }
    });
    Some(tx)
}

/// A short name for a child on the parent's status card: its mission's first
/// line, cut to a phone-width length.
pub fn child_status_label(mission: &str) -> String {
    const MAX_CHARS: usize = 60;
    let line = mission.lines().next().unwrap_or("").trim();
    let line = if line.is_empty() { "Sub-agent" } else { line };
    let is_path = |word: &str| word.contains('/') || word.contains('\\') || word.starts_with('~');
    if line.chars().count() <= MAX_CHARS && !line.split_whitespace().any(is_path) {
        return line.to_string();
    }
    // Whole words up to the limit, stopping before a path: a label never
    // ends mid-word or mid-path (live 2026-11-13: "… in exactly
    // /Users/davidloor/projects/cla").
    let mut label = String::new();
    for word in line.split_whitespace() {
        let longer = label.chars().count() + usize::from(!label.is_empty()) + word.chars().count();
        if is_path(word) || longer > MAX_CHARS - 1 {
            break;
        }
        if !label.is_empty() {
            label.push(' ');
        }
        label.push_str(word);
    }
    if label.is_empty() {
        let cut: String = line.chars().take(MAX_CHARS - 1).collect();
        return format!("{}…", cut.trim_end());
    }
    format!("{label}…")
}

/// What a stopped child had done, from its typed tool records: every call to
/// a tool not known to be read-only, finished or not. The child's session is
/// fresh per spawn, so its tool records are its whole run.
pub async fn child_work_report(
    event_store: &aidaemon_events::EventStore,
    child_session: &str,
    read_only_tools: &std::collections::HashSet<String>,
) -> ChildWorkReport {
    child_work_report_within(
        event_store,
        child_session,
        read_only_tools,
        MAX_REPORT_TOOL_RECORDS,
    )
    .await
}

/// Reads at most `max_records` tool records. When the run has more, the
/// earliest ones were not seen, so the report is `unknown`: it cannot show
/// that the child only read.
pub async fn child_work_report_within(
    event_store: &aidaemon_events::EventStore,
    child_session: &str,
    read_only_tools: &std::collections::HashSet<String>,
    max_records: usize,
) -> ChildWorkReport {
    let (events, saw_whole_run) = match event_store
        .query_events_by_types(
            child_session,
            &[EventType::ToolCall, EventType::ToolResult],
            max_records.saturating_add(1),
        )
        .await
    {
        Ok(mut events) => {
            let saw_whole_run = events.len() <= max_records;
            events.truncate(max_records);
            events.sort_by_key(|event| event.id);
            if !saw_whole_run {
                warn!(
                    child_session,
                    max_records, "A stopped child's tool records exceed the report window"
                );
            }
            (events, saw_whole_run)
        }
        Err(error) => {
            warn!(child_session, %error, "Could not read a stopped child's tool records");
            return ChildWorkReport {
                unknown: true,
                ..Default::default()
            };
        }
    };
    let results: std::collections::HashMap<String, bool> = events
        .iter()
        .filter(|event| event.event_type == EventType::ToolResult)
        .filter_map(|event| event.parse_data::<aidaemon_events::ToolResultData>().ok())
        .map(|data| (data.tool_call_id, data.success))
        .collect();
    let mut report = ChildWorkReport {
        unknown: !saw_whole_run,
        ..Default::default()
    };
    for call in events
        .iter()
        .filter(|event| event.event_type == EventType::ToolCall)
        .filter_map(|event| event.parse_data::<aidaemon_events::ToolCallData>().ok())
        .filter(|call| !read_only_tools.contains(&call.name))
    {
        let detail = call.summary.unwrap_or_else(|| call.arguments.to_string());
        let detail: String = detail.chars().take(REPORTED_ACTION_CHARS).collect();
        let entry = format!("{}: {detail}", call.name);
        match results.get(&call.tool_call_id) {
            Some(true) => report.changed.push(entry),
            Some(false) => report.changed.push(format!("{entry} (reported a failure)")),
            None => report.unfinished.push(entry),
        }
    }
    for list in [&mut report.changed, &mut report.unfinished] {
        if list.len() > MAX_REPORTED_ACTIONS {
            let omitted = list.len() - MAX_REPORTED_ACTIONS;
            list.drain(..omitted);
            list.insert(0, format!("(and {omitted} earlier)"));
        }
    }
    report
}

/// The specialist the caller named on a plain spawn, if it is one a caller
/// may name.
pub fn named_specialist(arg_specialist: Option<&str>) -> Option<SpecialistKind> {
    arg_specialist
        .and_then(SpecialistKind::from_str)
        .filter(|kind| *kind != SpecialistKind::TaskLead)
}

/// Whether a child gets its own `spawn_agent`. A named specialist is a leaf
/// (its allowlist would not otherwise cover the tool, which is added after
/// the allowlist is applied) unless the caller's explicit tool grant for the
/// child names `spawn_agent`: the caller asked for a coordinator.
pub fn child_can_spawn(
    child_depth: usize,
    max_depth: usize,
    role: AgentRole,
    named_specialist: bool,
    caller_granted_spawn: bool,
) -> bool {
    child_depth < max_depth
        && role != AgentRole::Executor
        && (!named_specialist || caller_granted_spawn)
}

/// The caller's explicit tool grant for the child includes `spawn_agent`.
pub fn caller_granted_spawn(parent: Option<&crate::authority::InheritedAuthority>) -> bool {
    parent
        .and_then(|parent| parent.authority.tools.as_ref())
        .is_some_and(|tools| tools.contains("spawn_agent"))
}

/// The system prompt of a plain child: the parent's prompt, the mission and,
/// when the caller named one, the specialist's own profile.
/// `can_spawn` is [`child_can_spawn`]'s decision, the one that also decides
/// whether the child gets `spawn_agent`: the prompt never states a rule of
/// its own about delegation.
pub fn plain_child_prompt(
    parent_prompt: &str,
    child_depth: usize,
    max_depth: usize,
    mission: &str,
    specialist_prompt: Option<&str>,
    can_spawn: bool,
) -> String {
    const NO_SPAWN: &str = "You CANNOT spawn further sub-agents; the `spawn_agent` tool is not \
         available to you. Complete the task directly.";
    let depth_note = if can_spawn {
        String::new()
    } else if child_depth >= max_depth {
        format!("\nYou are at the maximum sub-agent depth. {NO_SPAWN}")
    } else {
        format!("\n{NO_SPAWN}")
    };
    let mut prompt = format!(
        "{parent_prompt}\n\n## Sub-Agent Context\n\
         You are a sub-agent (depth {child_depth}/{max_depth}) spawned to accomplish a specific mission.\n\
         **Mission:** {mission}\n\n\
         Focus exclusively on this mission. Be concise. Return your findings/results \
         directly — they will be consumed by the parent agent. {own_duration}{depth_note}",
        own_duration = crate::specialists::compose::OWN_DURATION_RULE,
    );
    if let Some(specialist_prompt) = specialist_prompt {
        prompt.push_str("\n\n## Specialist Profile\n");
        prompt.push_str(specialist_prompt);
    }
    prompt
}

pub fn enforce_child_terminal_outcome(
    result: anyhow::Result<String>,
    task_end: Option<&TaskEndData>,
) -> anyhow::Result<String> {
    if result.is_ok()
        && task_end.is_some_and(|data| data.effective_outcome() == TaskOutcome::Failed)
    {
        let summary = task_end
            .and_then(|data| data.error.as_deref().or(data.summary.as_deref()))
            .unwrap_or("child task ended without a successful outcome");
        return Err(anyhow::anyhow!("Child task failed: {summary}"));
    }
    result
}

pub fn sqlite_busy_code(value: &str) -> bool {
    value
        .parse::<i32>()
        .ok()
        .is_some_and(|code| code & 0xff == 5)
}

pub fn message_has_sqlite_busy_code(message: &str) -> bool {
    message.split("code:").skip(1).any(|suffix| {
        let code = suffix
            .trim_start()
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>();
        sqlite_busy_code(&code)
    })
}

pub fn is_sqlite_busy(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if let Some(sqlx::Error::Database(database)) = cause.downcast_ref::<sqlx::Error>() {
            if database.code().is_some_and(|code| sqlite_busy_code(&code)) {
                return true;
            }
        }
        let message = cause.to_string().to_ascii_lowercase();
        message.contains("database is locked")
            || message.contains("database table is locked")
            || message.contains("sqlite_busy")
            || message_has_sqlite_busy_code(&message)
    })
}

pub fn worker_profile_id(kind: SpecialistKind) -> String {
    format!("profile-{}", kind.as_str().replace('_', "-"))
}

pub fn task_references_parent_context(task: &str) -> bool {
    let normalized = task.to_ascii_lowercase();
    [
        "completed task results",
        "prior knowledge",
        "parent context",
        "prompt context",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

/// Returns the task-lead execution-mode prose for the given `is_scheduled` flag.
///
/// This is the single source of truth for the two variants of the task-lead
/// execution mode paragraph. All three call sites — the legacy
/// `build_task_lead_prompt` builder, the registry-driven composition helper
/// `compose_task_lead_prompt_from_registry`, and the equivalence-test fixture
/// — must agree on this text or the byte-equivalence tests will fail.
pub fn task_lead_execution_mode(is_scheduled: bool) -> &'static str {
    if is_scheduled {
        "You have full tool access including `terminal`. For simple steps (single shell commands, \
         file writes), execute them directly. For complex multi-step work, you may still delegate \
         to executors via the workflow below."
    } else {
        "Your primary job is to plan and delegate work via executors or cli_agent. \
         However, you also have direct access to essential tools (read_file, write_file, \
         edit_file, terminal, search_files). Use delegation first, but if delegation fails \
         (cli_agent errors, spawn_agent blocked, executor failures), switch to direct \
         execution with your own tools rather than retrying broken delegation paths."
    }
}

// ---- The child-agent and conversation ports ----
//
// What `spawn_agent` and background completions need from the agent that
// runs them, so those tools need no concrete agent.

/// Result returned by a delegated child-agent run.
#[derive(Debug)]
pub struct ChildAgentRun {
    /// The child's own session.
    pub session: String,
    pub response: String,
    pub outcome: TaskOutcome,
    /// `Untrusted` when the child's run read (or started with) content
    /// someone other than the owner wrote.
    pub trust: aidaemon_core::SourceTrust,
    /// How its run ended ([`run_phase`]): what every surface shows.
    pub phase: crate::run_state::RunPhase,
    /// Its typed steps no receipt confirmed.
    pub evidence: crate::run_state::RunEvidence,
}

/// How long a child may go without progress (no model or tool call running
/// and no event of its own) before it is stopped. Never a limit on how long
/// it runs: a child that keeps working is bounded by its budgets.
#[derive(Debug, Clone, Copy)]
pub struct ChildTimeLimit {
    pub limit: std::time::Duration,
}

/// The child was stopped by the owner from the chat (`/cancel`, or a stop
/// request read as one).
#[derive(Debug)]
pub struct ChildStoppedByOwner;

impl std::fmt::Display for ChildStoppedByOwner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("stopped by the owner")
    }
}

impl std::error::Error for ChildStoppedByOwner {}

/// A child made no progress for its idle limit and was stopped.
#[derive(Debug)]
pub struct ChildTimedOut {
    pub limit_secs: u64,
    pub report: ChildWorkReport,
}

impl std::fmt::Display for ChildTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Specialist timed out: no progress for {} seconds",
            self.limit_secs
        )
    }
}

impl std::error::Error for ChildTimedOut {}

impl ChildTimedOut {
    /// What the parent needs to decide whether running it again is safe.
    pub fn render(&self) -> String {
        if !self.report.may_have_changed_something() {
            return format!(
                "The sub-agent made no progress for {} seconds and was stopped. It only \
                 used read-only tools, so running it again is safe.",
                self.limit_secs
            );
        }
        let mut text = format!(
            "The sub-agent made no progress for {} seconds and was stopped, but some of its \
             work may already have taken effect. Do NOT run it again until you have checked \
             the state of the items below: repeating them could duplicate side effects.",
            self.limit_secs
        );
        if self.report.unknown {
            text.push_str("\n\nIts action records could not be read in full.");
        }
        if !self.report.changed.is_empty() {
            text.push_str("\n\nActions it completed:");
            for item in &self.report.changed {
                text.push_str("\n- ");
                text.push_str(item);
            }
        }
        if !self.report.unfinished.is_empty() {
            text.push_str("\n\nStill running when it was stopped (outcome unknown):");
            for item in &self.report.unfinished {
                text.push_str("\n- ");
                text.push_str(item);
            }
        }
        text
    }
}

/// Owned input for a child-agent run. Using an owned request keeps the port
/// object-safe and safe to move into background tasks.
pub struct ChildAgentRequest {
    pub mission: String,
    pub task: String,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub channel_ctx: ChannelContext,
    pub user_role: UserRole,
    pub child_role: Option<AgentRole>,
    pub goal_id: Option<String>,
    pub task_id: Option<String>,
    pub project_scope: Option<String>,
    pub specialist: Option<String>,
    pub approval_session_id: Option<String>,
    /// The durable task attempt a task lead or executor was already claimed
    /// with; `None` lets the spawn find or claim its task's attempt.
    pub bound_task_attempt: Option<aidaemon_work::traits::TaskAttempt>,
    /// The spawning turn's actual authority, taint and root request
    /// (dispatcher-injected; `None` falls back to re-resolution).
    pub parent_authority: Option<crate::authority::InheritedAuthority>,
    /// `None` leaves the child unlimited.
    pub time_limit: Option<ChildTimeLimit>,
    pub model_tier: Option<aidaemon_providers::routing::Tier>,
}

impl ChildAgentRequest {
    /// A request for a child doing `task` within `mission`, with nothing
    /// else set.
    pub fn new(
        mission: impl Into<String>,
        task: impl Into<String>,
        channel_ctx: ChannelContext,
        user_role: UserRole,
    ) -> Self {
        Self {
            mission: mission.into(),
            task: task.into(),
            status_tx: None,
            channel_ctx,
            user_role,
            child_role: None,
            goal_id: None,
            task_id: None,
            project_scope: None,
            specialist: None,
            approval_session_id: None,
            bound_task_attempt: None,
            parent_authority: None,
            time_limit: None,
            model_tier: None,
        }
    }
}

/// Child-delegation capability required by `spawn_agent`.
#[async_trait]
pub trait ChildAgentRuntime: Send + Sync {
    fn depth(&self) -> usize;
    fn max_depth(&self) -> usize;
    fn role(&self) -> AgentRole;
    fn specialist_descriptions(&self) -> Vec<(String, String)>;

    /// The names of the tools this agent has, which bound what it can grant
    /// a child. `None`: not known (the grant is then checked only against
    /// the agent's own authority).
    fn tool_names(&self) -> Option<Vec<String>> {
        None
    }

    async fn validate_executor_task_for_spawn(
        &self,
        task_id: &str,
        expected_goal_id: Option<&str>,
    ) -> anyhow::Result<()>;

    async fn run_child(&self, request: ChildAgentRequest) -> anyhow::Result<ChildAgentRun>;

    /// The paths a child's declared scope names, as the file tools resolve
    /// them.
    async fn resolve_child_scope(&self, scope: Vec<String>) -> Vec<String> {
        scope
    }

    /// A token that stops the background child `id` when the owner cancels.
    async fn background_cancel_token(
        &self,
        _id: &str,
    ) -> Option<tokio_util::sync::CancellationToken> {
        None
    }

    /// Forget the token of a background child that has ended.
    async fn release_background_cancel_token(&self, _id: &str) {}

    /// The session a background child's result belongs to, given the session
    /// of the call that spawned it.
    fn background_delivery_session(&self, call_session: &str) -> String {
        call_session.to_string()
    }

    async fn salvage_executor_task_outcome(
        &self,
        task_id: &str,
        timeout_secs: u64,
    ) -> Option<SalvagedTaskOutcome>;

    async fn mark_executor_task_timeout(&self, task_id: &str, timeout_secs: u64);

    async fn deliver_background_child_result(
        &self,
        router: Option<&Weak<dyn OutboundRouter>>,
        parent_session_id: &str,
        text: &str,
        trust: aidaemon_core::SourceTrust,
    ) -> anyhow::Result<bool>;

    /// The run-state records of this agent tree's background helpers.
    fn helper_runs(&self) -> Arc<crate::run_state::RunStates>;
}

/// Owned input for re-entering an existing conversation.
pub struct ConversationRequest {
    pub session_id: String,
    pub user_text: String,
    pub status_tx: Option<mpsc::Sender<StatusUpdate>>,
    pub user_role: UserRole,
    pub channel_ctx: ChannelContext,
    pub heartbeat: Option<Arc<AtomicU64>>,
    /// Typed lifecycle edge back to the task whose background receipt caused
    /// this continuation.
    pub parent_task_id: Option<String>,
    pub parent_tool_call_id: Option<String>,
    pub parent_result_id: Option<String>,
}

/// Minimal agent surface needed by background command/CLI completion paths.
#[async_trait]
pub trait ConversationRuntime: Send + Sync {
    async fn continue_conversation(
        &self,
        request: ConversationRequest,
    ) -> anyhow::Result<AgentResponseEnvelope>;

    async fn record_continuation_delivery(
        &self,
        session_id: &str,
        delivery: aidaemon_events::ResponseDeliveryData,
    ) -> anyhow::Result<()>;
}

// ---- Building a child ----

/// What a parent asks its host to build for one child: its tools, model and
/// prompt, where it stands in the tree, and what it inherited.
pub struct ChildSpec {
    /// The child's tools (the host adds `spawn_agent` when `add_spawn_tool`).
    pub tools: Vec<Arc<dyn aidaemon_tools::OutcomeTool>>,
    pub model: String,
    pub system_prompt: String,
    pub depth: usize,
    pub role: AgentRole,
    pub task_id: Option<String>,
    pub goal_id: Option<String>,
    pub mandate_execution: Option<crate::turn_runtime::MandateExecutionFence>,
    pub cancel_token: Option<tokio_util::sync::CancellationToken>,
    /// The root agent's full tool list, for a task lead's executors.
    pub root_tools: Option<Vec<Arc<dyn aidaemon_tools::OutcomeTool>>>,
    pub add_spawn_tool: bool,
    pub inherited_project_scope: Option<String>,
    pub approval_session_id: Option<String>,
    pub max_iterations_override: Option<usize>,
    pub timeout_secs_override: Option<u64>,
    pub inherited_authority: crate::authority::InheritedAuthority,
    /// The child's progress clock: what keeps it from being stopped as idle.
    pub child_progress: Option<aidaemon_channels::owner_wait::ProgressClock>,
}

/// A child agent its host built: a turn host of its own that runs turns.
#[async_trait]
pub trait ChildAgent: crate::host::TurnHost {
    /// Run one turn of the child.
    async fn handle_message(
        &self,
        session_id: &str,
        user_text: &str,
        status_tx: Option<mpsc::Sender<StatusUpdate>>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        heartbeat: Option<Arc<AtomicU64>>,
    ) -> anyhow::Result<String>;

    /// Set once any turn of the child read (or started with) untrusted
    /// content, even when the taint could not be recorded.
    fn read_untrusted(&self) -> Arc<std::sync::atomic::AtomicBool>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run_state::RunFailure;
    use aidaemon_events::RunAggregate;

    /// A run's aggregate with checklist steps `(state, required effect)` and
    /// operations `(outcome, effects)`.
    fn aggregate(steps: &[(&str, u32)], operations: &[(&str, u32)]) -> RunAggregate {
        let obligations: serde_json::Map<String, serde_json::Value> = steps
            .iter()
            .enumerate()
            .map(|(index, (state, effect))| {
                let id = format!("task:t/obligation:checklist:{index}");
                (
                    id.clone(),
                    serde_json::json!({
                        "id": id,
                        "class": if *effect == 0 { "observe" } else { "achieve" },
                        "state": state,
                        "required_effect": effect,
                        "summary": format!(
                            "checklist item {}: step {index} — needs a succeeded receipt \
                             carrying local_workspace_write|process_state",
                            index + 1
                        ),
                        "step": format!("step {index}"),
                    }),
                )
            })
            .collect();
        let operations: serde_json::Map<String, serde_json::Value> = operations
            .iter()
            .enumerate()
            .map(|(index, (outcome, effects))| {
                let id = format!("op{index}");
                (
                    id.clone(),
                    serde_json::json!({
                        "operation_id": id,
                        "tool_name": "terminal",
                        "outcome": outcome,
                        "dispatched": true,
                        "mutation_effects": effects,
                        "result_id": format!("sha256:{index}"),
                    }),
                )
            })
            .collect();
        serde_json::from_value(serde_json::json!({
            "schema_version": 1,
            "task_id": "t",
            "obligations": obligations,
            "operations": operations,
        }))
        .unwrap()
    }

    fn checklist(aggregate: &RunAggregate) -> Vec<&aidaemon_events::RunObligation> {
        aggregate.obligations.values().collect()
    }

    /// The live run (2026-11-13): it looked at its folder (a satisfied
    /// observation), changed nothing, and set every step that needed a
    /// change aside. Its outcome was only "partial"; its run failed.
    #[test]
    fn a_run_that_changed_nothing_and_did_no_change_step_failed() {
        let run = aggregate(
            &[
                ("satisfied", 0),
                ("deferred", 130),
                ("deferred", 0),
                ("deferred", 64),
            ],
            &[("failed_permanent", 0), ("succeeded", 0), ("succeeded", 0)],
        );
        assert_eq!(
            nothing_done(TaskOutcome::Partial, &checklist(&run), &run),
            Some(RunFailure::NothingDone {
                undone: vec!["step 1".to_string(), "step 3".to_string()],
            })
        );
    }

    /// Any change made, any change step done, a success, or no step that
    /// needed a change: not a failure of this kind.
    #[test]
    fn a_run_that_did_some_of_its_work_did_not_fail_for_nothing_done() {
        let changed = aggregate(&[("deferred", 2)], &[("succeeded", 2)]);
        assert_eq!(
            nothing_done(TaskOutcome::Partial, &checklist(&changed), &changed),
            None
        );
        let step_done = aggregate(&[("satisfied", 2), ("deferred", 64)], &[]);
        assert_eq!(
            nothing_done(TaskOutcome::Partial, &checklist(&step_done), &step_done),
            None
        );
        let looked_only = aggregate(&[("deferred", 0)], &[("succeeded", 0)]);
        assert_eq!(
            nothing_done(TaskOutcome::Partial, &checklist(&looked_only), &looked_only),
            None
        );
        let undone = aggregate(&[("deferred", 2)], &[]);
        assert_eq!(
            nothing_done(TaskOutcome::Succeeded, &checklist(&undone), &undone),
            None
        );
    }

    /// A run whose steps no receipt confirmed names them in their own
    /// words: the owner never reads the internal demand ("needs a succeeded
    /// receipt carrying local_workspace_write|process_state").
    #[test]
    fn unconfirmed_steps_are_named_in_their_own_words() {
        let run = aggregate(&[("pending", 2), ("pending", 64)], &[("succeeded", 2)]);
        let mut run = run;
        run.executor_claimed_outcome = Some(aidaemon_events::ExecutorClaimedOutcome::Done);
        let end: aidaemon_events::TaskEndData = serde_json::from_value(serde_json::json!({
            "task_id": "t",
            "status": "completed",
            "duration_secs": 1,
            "iterations": 1,
            "tool_calls_count": 1,
        }))
        .unwrap();
        let (phase, evidence) = run_phase(TaskOutcome::Partial, Some(&end), Some(&run));
        assert_eq!(phase, crate::run_state::RunPhase::Succeeded);
        let line = evidence.line().expect("an evidence line");
        assert_eq!(line, "I couldn't verify 2 of its steps: step 0; step 1.");
        for internal in [
            "receipt",
            "local_workspace_write",
            "process_state",
            "checklist item",
        ] {
            assert!(!line.contains(internal), "{line}");
        }
    }

    /// A long label stops at a whole word, before any path: never a cut
    /// path or a cut word.
    #[test]
    fn a_label_never_ends_in_a_cut_path() {
        let label = child_status_label(
            "Install a fresh Drupal site with demo content in exactly /Users/someone/projects/claude",
        );
        assert_eq!(
            label,
            "Install a fresh Drupal site with demo content in exactly…"
        );
        assert_eq!(child_status_label("Count to 30"), "Count to 30");
    }

    use aidaemon_core::card::ActivityKind;

    fn rendered_checklist(done: usize, total: usize) -> String {
        let steps = (0..total).map(|i| format!("synthetic step {i}")).collect();
        let mut plan = aidaemon_work::plans::TaskPlan::new(
            "synthetic-session",
            "synthetic-task",
            "description",
            steps,
            "track_requirements",
        );
        for step in plan.steps.iter_mut().take(done) {
            step.status = aidaemon_work::plans::StepStatus::Completed;
        }
        if done == total {
            plan.status = aidaemon_work::plans::PlanStatus::Completed;
        } else {
            plan.current_step = done;
        }
        plan.render_compact_checklist()
    }

    /// Two parallel children each declare a checklist. Neither reaches the
    /// parent's card (which would replace the parent's plan); each step
    /// becomes its own child's record with a typed kind, a step handed to
    /// the background included, and a tool ending is not a step.
    #[tokio::test]
    async fn a_childs_checklist_never_replaces_the_parents_plan() {
        let (parent_tx, mut parent_rx) = mpsc::channel(16);
        let wordpress = child_status_sender(
            Some(parent_tx.clone()),
            "Build the WordPress demo site for Acme Corp",
            false,
            None,
        )
        .unwrap();
        let drupal = child_status_sender(
            Some(parent_tx),
            "Build the Drupal demo site for Acme Corp",
            false,
            None,
        )
        .unwrap();
        wordpress
            .send(StatusUpdate::Checklist(aidaemon_core::card::PlanCard {
                text: rendered_checklist(5, 5),
                steps: 5,
            }))
            .await
            .unwrap();
        drupal
            .send(StatusUpdate::Checklist(aidaemon_core::card::PlanCard {
                text: rendered_checklist(1, 4),
                steps: 4,
            }))
            .await
            .unwrap();
        drupal
            .send(StatusUpdate::ToolStart {
                name: "running the site installer".into(),
                summary: String::new(),
                kind: ActivityKind::Run,
            })
            .await
            .unwrap();
        drupal
            .send(StatusUpdate::ToolComplete {
                name: "running the site installer".into(),
                summary: "done".into(),
            })
            .await
            .unwrap();
        drupal
            .send(StatusUpdate::StepHandedOff {
                kind: ActivityKind::Run,
            })
            .await
            .unwrap();
        drop((wordpress, drupal));

        let mut lines = Vec::new();
        while let Some(update) = parent_rx.recv().await {
            match update {
                StatusUpdate::Checklist(_) => panic!("child checklist reached the parent card"),
                StatusUpdate::Helper(view) => {
                    assert!(!view.background);
                    lines.push(format!(
                        "{}: {}",
                        view.label,
                        view.step.map(|step| step.words()).unwrap_or_default()
                    ));
                }
                other => panic!("unexpected update {other:?}"),
            }
        }
        assert_eq!(
            lines,
            vec![
                "Build the Drupal demo site for Acme Corp: Running a command".to_string(),
                "Build the Drupal demo site for Acme Corp: Running a command · goes on in the \
                 background"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn running_siblings_with_one_title_are_told_apart_by_their_task() {
        let registry = Arc::new(RunningChildren::default());
        let mission = "Run one shell command and report its exact output.";
        let _alpha = registry.enter(
            "chat-1",
            mission,
            "Run this exact shell command: sleep 120 && echo alpha-test . Report the output.",
            "child-alpha",
            None,
        );
        let _beta = registry.enter(
            "chat-1",
            mission,
            "Run this exact shell command: sleep 120 && echo beta-test . Report the output.",
            "child-beta",
            None,
        );
        let _other_chat = registry.enter("chat-2", mission, "Run: date", "child-other", None);
        let labels: Vec<String> = registry
            .for_session("chat-1")
            .into_iter()
            .map(|child| child.label)
            .collect();
        assert_eq!(
            labels,
            vec![
                format!("{mission} (…echo alpha-test . Report the…)"),
                format!("{mission} (…echo beta-test . Report the…)"),
            ]
        );
        let alone = registry.for_session("chat-2");
        assert_eq!(alone[0].label, mission);
        drop(_beta);
        assert_eq!(registry.for_session("chat-1")[0].label, mission);
    }

    /// A child is listed under the chat turn that started it, and a stop
    /// from the chat reaches it by its key.
    #[test]
    fn a_listed_child_carries_its_request_and_can_be_stopped() {
        let registry = Arc::new(RunningChildren::default());
        let turn = registry.begin_request("chat-1", "task-a");
        let child = registry.enter("chat-1", "Counting agent A", "count", "child-a", None);
        drop(turn);
        let later = registry.enter("chat-1", "Counting agent B", "count", "child-b", None);
        let listed = registry.listed("chat-1");
        assert_eq!(listed[0].request.as_deref(), Some("task-a"));
        assert_eq!(listed[1].request, None);
        assert_eq!(registry.current_request("chat-1"), None);
        let stop = child.stopped();
        assert_eq!(registry.stop(&listed[0].key).as_deref(), Some("child-a"));
        assert!(stop.is_cancelled());
        assert!(registry.is_listed(&listed[0].key));
        drop(child);
        assert!(!registry.is_listed(&listed[0].key));
        assert!(registry.stop(&listed[0].key).is_none());
        drop(later);
    }

    #[test]
    fn a_child_status_label_fits_a_phone() {
        assert_eq!(child_status_label("  \n"), "Sub-agent");
        let long = "x".repeat(200);
        let label = child_status_label(&long);
        assert_eq!(label.chars().count(), 60);
        assert!(label.ends_with('…'));
    }

    async fn synthetic_event_store() -> aidaemon_events::EventStore {
        let db_file = tempfile::NamedTempFile::new().expect("temp db file");
        let path = db_file.into_temp_path().keep().expect("keep temp db path");
        let pool = sqlx::SqlitePool::connect(&format!("sqlite:{}", path.display()))
            .await
            .expect("connect sqlite");
        aidaemon_events::EventStore::on_paths(
            pool,
            aidaemon_tools::LexicalPaths::daemon_host(),
            Arc::new(aidaemon_events::NoEventHost),
        )
        .await
        .expect("init event store")
    }

    async fn record_tool_call(
        store: &aidaemon_events::EventStore,
        session: &str,
        id: &str,
        name: &str,
        finished: bool,
    ) {
        store
            .append(aidaemon_events::Event::new(
                session,
                EventType::ToolCall,
                serde_json::json!({ "tool_call_id": id, "name": name, "arguments": {} }),
            ))
            .await
            .expect("append tool call");
        if finished {
            store
                .append(aidaemon_events::Event::new(
                    session,
                    EventType::ToolResult,
                    serde_json::json!({
                        "tool_call_id": id,
                        "name": name,
                        "result": "ok",
                        "success": true,
                        "duration_ms": 1,
                    }),
                ))
                .await
                .expect("append tool result");
        }
    }

    /// An early change stays in the report however many read-only calls
    /// follow it.
    #[tokio::test]
    async fn a_stopped_childs_report_covers_its_whole_run() {
        let store = synthetic_event_store().await;
        let session = "specialist:executor:synthetic-1";
        let read_only: std::collections::HashSet<String> =
            ["read_file".to_string()].into_iter().collect();
        record_tool_call(&store, session, "call-write", "write_file", true).await;
        for n in 0..300 {
            record_tool_call(
                &store,
                session,
                &format!("call-read-{n}"),
                "read_file",
                true,
            )
            .await;
        }
        let report = child_work_report(&store, session, &read_only).await;
        assert!(report.may_have_changed_something());
        assert_eq!(report.changed.len(), 1);
        assert!(!report.unknown);
    }

    /// A report that cannot show it saw the run from its first record does
    /// not claim the child only read.
    #[tokio::test]
    async fn a_report_that_cannot_see_the_whole_run_is_unknown() {
        let store = synthetic_event_store().await;
        let session = "specialist:executor:synthetic-2";
        let read_only: std::collections::HashSet<String> =
            ["read_file".to_string()].into_iter().collect();
        for n in 0..6 {
            record_tool_call(
                &store,
                session,
                &format!("call-read-{n}"),
                "read_file",
                true,
            )
            .await;
        }
        let report = child_work_report_within(&store, session, &read_only, 10).await;
        assert!(report.unknown);
        assert!(report.may_have_changed_something());
        let report = child_work_report_within(&store, session, &read_only, 13).await;
        assert!(!report.unknown);
        assert!(!report.may_have_changed_something());
    }

    /// A child's end record is found however many records follow it.
    #[tokio::test]
    async fn a_childs_end_record_is_found_behind_later_records() {
        let store = synthetic_event_store().await;
        let session = "specialist:executor:synthetic-3";
        store
            .append(aidaemon_events::Event::new(
                session,
                EventType::TaskEnd,
                serde_json::json!({
                    "task_id": "task-synthetic-3",
                    "status": "completed",
                    "outcome": "succeeded",
                }),
            ))
            .await
            .expect("append task end");
        for n in 0..40 {
            record_tool_call(
                &store,
                session,
                &format!("call-late-{n}"),
                "read_file",
                false,
            )
            .await;
        }
        let record = latest_child_task_end(&store, session).await;
        assert_eq!(resolve_child_outcome(true, &record), TaskOutcome::Succeeded);
    }

    #[test]
    fn a_child_with_a_refused_write_it_never_made_is_incomplete() {
        assert_eq!(
            settle_coordinated_outcome(TaskOutcome::Succeeded, true),
            TaskOutcome::Partial
        );
        assert_eq!(
            settle_coordinated_outcome(TaskOutcome::Succeeded, false),
            TaskOutcome::Succeeded
        );
        assert_eq!(
            settle_coordinated_outcome(TaskOutcome::Failed, true),
            TaskOutcome::Failed
        );
    }

    #[test]
    fn a_reply_with_no_readable_end_record_is_not_a_success() {
        let unreadable: anyhow::Result<Option<TaskEndData>> =
            Err(anyhow::anyhow!("database is locked"));
        assert_eq!(
            resolve_child_outcome(true, &unreadable),
            TaskOutcome::Partial
        );
        assert_eq!(resolve_child_outcome(true, &Ok(None)), TaskOutcome::Partial);
        assert_eq!(resolve_child_outcome(false, &Ok(None)), TaskOutcome::Failed);
    }

    /// The child's prompt says what `child_can_spawn` decided: a specialist
    /// its caller made a coordinator is not told it cannot delegate.
    #[test]
    fn the_child_prompt_follows_the_spawn_decision() {
        let coordinator = plain_child_prompt("P", 1, 3, "Coordinate", Some("profile"), true);
        assert!(!coordinator.contains("CANNOT spawn"));
        let leaf = plain_child_prompt("P", 1, 3, "Count", Some("profile"), false);
        assert!(leaf.contains("CANNOT spawn"));
        let deepest = plain_child_prompt("P", 3, 3, "Count", None, false);
        assert!(deepest.contains("maximum sub-agent depth"));
    }

    #[test]
    fn a_named_specialist_is_a_leaf_and_a_plain_child_is_not() {
        assert!(child_can_spawn(1, 3, AgentRole::Orchestrator, false, false));
        assert!(!child_can_spawn(1, 3, AgentRole::Orchestrator, true, false));
        assert!(!child_can_spawn(
            3,
            3,
            AgentRole::Orchestrator,
            false,
            false
        ));
        assert!(!child_can_spawn(1, 3, AgentRole::Executor, false, false));
        // A caller that grants `spawn_agent` asked for a coordinator, within
        // the depth limit and never for an executor.
        assert!(child_can_spawn(1, 3, AgentRole::Orchestrator, true, true));
        assert!(!child_can_spawn(3, 3, AgentRole::Orchestrator, true, true));
        assert!(!child_can_spawn(1, 3, AgentRole::Executor, true, true));
    }

    #[test]
    fn task_lead_cannot_be_named_on_a_plain_spawn() {
        assert_eq!(named_specialist(Some("task_lead")), None);
        assert_eq!(named_specialist(Some("not-a-kind")), None);
        assert_eq!(named_specialist(None), None);
        assert_eq!(
            named_specialist(Some("review")),
            Some(SpecialistKind::Review)
        );
    }

    /// A child that saw untrusted content but could not record it, or
    /// whose record cannot be read, returns an untrusted result.
    #[test]
    fn a_childs_unrecorded_or_unreadable_taint_fails_closed() {
        use aidaemon_core::SourceTrust::{Trusted, Untrusted};
        assert_eq!(child_result_trust(false, &Ok(Vec::new())), Trusted);
        assert_eq!(child_result_trust(true, &Ok(Vec::new())), Untrusted);
        assert_eq!(
            child_result_trust(false, &Ok(vec!["turn-a".to_string()])),
            Untrusted
        );
        assert_eq!(
            child_result_trust(false, &Err(anyhow::anyhow!("unreadable"))),
            Untrusted
        );
    }

    #[test]
    fn sqlite_busy_retry_classifier_is_narrow() {
        assert!(is_sqlite_busy(&anyhow::anyhow!(
            "error returned from database: (code: 5) database is locked"
        )));
        assert!(is_sqlite_busy(&anyhow::anyhow!("SQLITE_BUSY")));
        for code in [261, 517, 773] {
            assert!(is_sqlite_busy(&anyhow::anyhow!(
                "error returned from database: (code: {code}) database is locked"
            )));
        }
        assert!(!is_sqlite_busy(&anyhow::anyhow!(
            "error returned from database: (code: 516) unrelated failure"
        )));
        assert!(!is_sqlite_busy(&anyhow::anyhow!("executor lease was lost")));
    }

    /// A helper never states its own run's duration: the owner reads only
    /// the run record's measured times, under its result.
    #[test]
    fn a_helper_is_told_its_run_time_is_the_daemons() {
        let rule = crate::specialists::compose::OWN_DURATION_RULE;
        assert!(plain_child_prompt("P", 1, 3, "Count", None, false).contains(rule));
        assert!(plain_child_prompt("P", 1, 3, "Coordinate", Some("profile"), true).contains(rule));
    }
}
