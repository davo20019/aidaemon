//! Spawning sub-agents: the running-children registry `/status` and `/tasks`
//! read, how a child's progress and end reach its parent, what a stopped
//! child had done, and the plain child's prompt. Building and running the
//! child itself is the host's.

use std::sync::Arc;
use std::time::Duration;

use aidaemon_agents::specialist::SpecialistKind;
use aidaemon_events::traits::AgentRole;
use aidaemon_events::{EventType, TaskEndData, TaskOutcome};
use aidaemon_tools::status::StatusUpdate;
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
    /// The child's own typed verdict was `done`, and its outcome is partial
    /// only because no receipt confirmed some of its listed steps
    /// ([`reported_done_unconfirmed`]).
    pub reported_done: bool,
}

/// Whether a child's partial outcome is only for want of receipts: it
/// ended its run normally with its own typed verdict `done`, nothing it
/// started still runs, the owner declined nothing, and it set none of its
/// listed steps aside (failed, skipped or deferred); only some steps were
/// never matched by a receipt. The owner is then told it finished.
pub fn reported_done_unconfirmed(
    outcome: TaskOutcome,
    end: &TaskEndData,
    aggregate: &aidaemon_events::RunAggregate,
) -> bool {
    use aidaemon_events::RunObligationState;
    outcome == TaskOutcome::Partial
        && end.status == aidaemon_events::TaskStatus::Completed
        && aggregate.executor_claimed_outcome == Some(aidaemon_events::ExecutorClaimedOutcome::Done)
        && !aggregate.has_background_in_flight()
        && !aggregate.owner_declined()
        && aggregate
            .obligations
            .iter()
            .filter(|(id, _)| id.contains("/obligation:checklist:"))
            .all(|(_, obligation)| {
                matches!(
                    obligation.state,
                    RunObligationState::Satisfied | RunObligationState::Pending
                )
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
}

/// A running child's latest step, written by its status channel.
pub type ChildStep = Arc<std::sync::Mutex<Option<String>>>;

/// A listed child, as the chat's running work shows it.
pub struct ListedChildView {
    pub key: String,
    pub child: aidaemon_channels::ports::RunningChild,
    pub request: Option<String>,
    /// Its latest step, as `/tasks` and the status reply give it.
    pub step: Option<String>,
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
    pub fn enter(
        self: &Arc<Self>,
        session_id: &str,
        mission: &str,
        task: &str,
        child_session: &str,
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
                ListedChildView {
                    key: child_key(child.id),
                    child: aidaemon_channels::ports::RunningChild {
                        label,
                        started_at: child.started_at,
                    },
                    request: child.request.clone(),
                    step: child
                        .step
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone(),
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
/// reaches the card as one `SubAgentProgress` line naming that child instead of
/// replacing the parent's plan (two parallel children used to overwrite it in
/// turn, leaving whichever wrote last looking like the whole task). Each of
/// the child's steps (a tool starting or ending) is likewise that child's
/// progress line, so the parent's card and its status reply say what each
/// helper is doing ("Drupal install: running the site installer"), and so
/// is a step of the child's that goes on in the background. Every other
/// update passes through unchanged.
pub fn child_status_sender(
    parent: Option<mpsc::Sender<StatusUpdate>>,
    mission: &str,
    step: Option<ChildStep>,
) -> Option<mpsc::Sender<StatusUpdate>> {
    let parent = parent?;
    let label = child_status_label(mission);
    let (tx, mut rx) = mpsc::channel::<StatusUpdate>(64);
    tokio::spawn(async move {
        while let Some(update) = rx.recv().await {
            let update = match update {
                StatusUpdate::Checklist { text } => {
                    let Some(progress) = aidaemon_work::plans::checklist_progress(&text) else {
                        continue;
                    };
                    StatusUpdate::SubAgentProgress {
                        label: label.clone(),
                        progress,
                    }
                }
                StatusUpdate::ToolStart { name, summary } => StatusUpdate::SubAgentProgress {
                    label: label.clone(),
                    progress: if summary.trim().is_empty() {
                        name
                    } else {
                        format!("{name}: {summary}")
                    },
                },
                StatusUpdate::ToolComplete { name, .. } => StatusUpdate::SubAgentProgress {
                    label: label.clone(),
                    progress: aidaemon_channels::live_status::step_ended_line(&name),
                },
                // A child's step that went on in the background is that
                // child's progress, never the parent's own step (live
                // 2026-11-13: "… goes on in the background" stood on the
                // card as if the turn itself had handed off).
                StatusUpdate::StepHandedOff { name, to_helper } => StatusUpdate::SubAgentProgress {
                    label: label.clone(),
                    progress: aidaemon_channels::live_status::helper_step_handed_off(
                        &name, to_helper,
                    ),
                },
                // What the child's running command printed last is that
                // child's progress line, as it runs.
                StatusUpdate::CommandOutput { line, .. } => StatusUpdate::SubAgentProgress {
                    label: label.clone(),
                    progress: format!("running a command: {line}"),
                },
                other => other,
            };
            // Its own latest step, for `/tasks`.
            if let (
                Some(slot),
                StatusUpdate::SubAgentProgress {
                    label: of,
                    progress,
                },
            ) = (step.as_ref(), &update)
            {
                if *of == label {
                    *slot
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        Some(progress.clone());
                }
            }
            if parent.send(update).await.is_err() {
                break;
            }
        }
    });
    Some(tx)
}

/// How a child's progress line on the parent's card reads once it ended.
pub fn child_end_progress(outcome: TaskOutcome) -> &'static str {
    match outcome {
        TaskOutcome::Succeeded => "done",
        TaskOutcome::Partial => "stopped short",
        TaskOutcome::Failed => "didn't finish",
    }
}

/// A short name for a child on the parent's status card: its mission's first
/// line, cut to a phone-width length.
pub fn child_status_label(mission: &str) -> String {
    const MAX_CHARS: usize = 60;
    let line = mission.lines().next().unwrap_or("").trim();
    let line = if line.is_empty() { "Sub-agent" } else { line };
    if line.chars().count() <= MAX_CHARS {
        return line.to_string();
    }
    let cut: String = line.chars().take(MAX_CHARS - 1).collect();
    format!("{}…", cut.trim_end())
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
         directly — they will be consumed by the parent agent.{depth_note}"
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
