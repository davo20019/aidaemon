//! A conversation: its transcript (Markdown replies, streamed as they
//! arrive), the approvals it waits on, and the composer. A new
//! conversation is the same page with no session yet.

use aidaemon_client_core::state::Line;
use aidaemon_client_protocol::{
    AttachmentRef, MidRun, RunPhase, RunSummary, ServedBy, ToolCallInfo,
};
use nsh_base::{
    MarkdownOptions, TextViewLinks, TextViewPart, TextViewParts, button, input, text_view,
};
use nsh_core::platform::Idiom;
use nsh_core::prelude::*;

use super::approvals;
use crate::store::{ActivityMode, DraftFileState, Felt, Store};
use crate::ui::{self, Glyph, Size, Variant, colors, radius, size, type_scale, working};

/// A row of the transcript.
#[derive(Debug, Clone, PartialEq)]
enum Row {
    /// A day break: "Today", "Yesterday", "Mar 3".
    Day(String),
    User(String, u64, Vec<AttachmentRef>),
    /// A reply: its text, time and files, whether it only says the work
    /// went on in the background, whether its time and Copy show (the
    /// last reply before the owner speaks again), and the provider and
    /// model that wrote it.
    Assistant(
        String,
        u64,
        Vec<AttachmentRef>,
        bool,
        bool,
        Option<ServedBy>,
    ),
    Note(String),
    /// The reply being written and what the agent is doing.
    Live,
    Approval(String),
    /// The tools a reply's turn used, how many seconds it took, and what
    /// each was given and gave back as far as known.
    Activity(Vec<String>, u64, Vec<ToolCallInfo>),
    /// The last turn failed or was stopped: send its message again.
    Retry,
    /// Work handed to the background runs; its report comes here.
    Background,
    /// The runs one request started (`RunSummary.group`, or a run alone:
    /// its id), under the reply that started them; `headed`: with their
    /// own heading (the running turn's line heads them while it runs).
    Runs {
        key: String,
        headed: bool,
    },
}

impl Row {
    fn key(&self, index: usize) -> String {
        match self {
            Self::Live => "live".into(),
            Self::Retry => "retry".into(),
            Self::Background => "background".into(),
            Self::Approval(id) => format!("approval-{id}"),
            Self::Runs { key, headed } => format!("runs-{key}-{headed}"),
            Self::Day(label) => format!("day-{label}-{index}"),
            _ => format!("line-{index}"),
        }
    }
}

fn rows(store: Store, session: &str) -> Vec<Row> {
    let now = crate::store::now_ms();
    let mode = store.activity_mode();
    let mut rows: Vec<Row> = Vec::new();
    let mut last_day = None;
    for line in store.lines(session) {
        let at = match &line {
            Line::User { at_ms, .. } | Line::Assistant { at_ms, .. } => *at_ms,
            Line::Note { .. } | Line::Activity { .. } => 0,
        };
        if at > 0 {
            let day = ui::time::day(at);
            if last_day != Some(day) {
                last_day = Some(day);
                rows.push(Row::Day(ui::time::day_label(at, now)));
            }
        }
        rows.push(match line {
            Line::User {
                text,
                at_ms,
                attachments,
                ..
            } => Row::User(text, at_ms, attachments),
            Line::Assistant {
                text,
                at_ms,
                attachments,
                handed_off,
                served_by,
            } => Row::Assistant(text, at_ms, attachments, handed_off, false, served_by),
            Line::Note { text, .. } => Row::Note(text),
            Line::Activity { .. } if mode == ActivityMode::Hidden => continue,
            Line::Activity { tools, secs, calls } => Row::Activity(tools, secs, calls),
        });
    }
    // A run of replies shows its time and Copy once, under the last; a
    // launch line never does (its report is the reply that matters).
    let mut last_of_run = true;
    for row in rows.iter_mut().rev() {
        match row {
            Row::User(..) => last_of_run = true,
            Row::Assistant(_, _, _, handed_off, footer, _) if !*handed_off => {
                *footer = last_of_run;
                last_of_run = false;
            }
            _ => {}
        }
    }
    // Each request's runs sit under the reply that started them, and stay
    // there once they end: which agents ran and how each ended. A request
    // still being answered has its runs under its live line.
    let tail = place_runs(&mut rows, &store.runs_for(session));
    let mut rows = without_launch_lines(rows);
    let live = store.live(session).is_some();
    let tail = tail.into_iter().map(|key| Row::Runs { key, headed: !live });
    if live {
        rows.push(Row::Live);
        rows.extend(tail);
    } else if store.in_background(session)
        && !matches!(rows.last(), Some(Row::Assistant(_, _, _, true, ..)))
    {
        // A launch line already says it; the work's own rows show it
        // running, with Stop.
        rows.extend(tail);
        rows.push(Row::Background);
    } else {
        rows.extend(tail);
        if store.can_retry(session) {
            rows.push(Row::Retry);
        }
    }
    rows.extend(
        store
            .approvals_for(session)
            .into_iter()
            .map(|a| Row::Approval(a.id)),
    );
    rows
}

/// `rows` without each launch line ("Started in the background: …") that
/// sits over its own runs' rows: the rows say it, live.
fn without_launch_lines(rows: Vec<Row>) -> Vec<Row> {
    let mut kept = Vec::with_capacity(rows.len());
    let mut iter = rows.into_iter().peekable();
    while let Some(row) = iter.next() {
        let launch_over_runs = matches!(row, Row::Assistant(_, _, _, true, ..))
            && matches!(iter.peek(), Some(Row::Runs { .. }));
        if !launch_over_runs {
            kept.push(row);
        }
    }
    kept
}

/// A run's group: the request that started it, or the run alone.
fn run_key(run: &RunSummary) -> String {
    run.group.clone().unwrap_or_else(|| run.id.clone())
}

/// Places each request's runs under the first reply written at or after
/// the first of them started (the reply that started them); returns, in
/// order, the groups no reply follows yet (their request is still being
/// answered, or its reply has not come).
fn place_runs(rows: &mut Vec<Row>, runs: &[RunSummary]) -> Vec<String> {
    let mut groups: Vec<(String, u64)> = Vec::new();
    for run in runs {
        let key = run_key(run);
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, at)) => *at = (*at).min(run.started_at_ms),
            None => groups.push((key, run.started_at_ms)),
        }
    }
    groups.sort_by_key(|(_, at)| *at);
    let mut tail = Vec::new();
    // From the last group back, so each insert leaves earlier places as
    // they were.
    for (key, at) in groups.into_iter().rev() {
        let reply = rows
            .iter()
            .position(|row| matches!(row, Row::Assistant(_, reply_at, ..) if *reply_at >= at));
        match reply {
            Some(index) => rows.insert(index + 1, Row::Runs { key, headed: true }),
            None => tail.insert(0, key),
        }
    }
    tail
}

/// The page for `session` (`None`: a new conversation).
pub fn page(store: Store, session: Option<String>) -> View {
    let body = match &session {
        Some(id) => transcript(store, id.clone()).into_element(),
        None => welcome(store).into_element(),
    };
    view()
        .test_id("conversation")
        .flex_1()
        .min_w(px(0.0))
        .child(header(store, session.clone()))
        .child(view().flex_1().child(body))
        .child(offers(store, session.clone()))
        .child(notices(store))
        .child(running_work(store, session.clone()))
        .child(match session.clone() {
            // A group hears what is sent from here: the owner says so first.
            Some(id) if untracked(|| store.is_group(&id)) => {
                let gate = id.clone();
                when(move || store.may_post(&gate), {
                    let session = session.clone();
                    move || composer(store, session.clone())
                })
                .otherwise(move || group_gate(store, id.clone()))
                .into_element()
            }
            _ => composer(store, session).into_element(),
        })
}

/// An older daemon's background work, over the composer: its task list.
/// A daemon with run records (protocol 1.46) has its runs in the
/// transcript, under the reply that started them.
fn running_work(store: Store, session: Option<String>) -> View {
    let Some(session) = session else {
        return view();
    };
    let has_runs = {
        let session = session.clone();
        computed(move || !store.runs_for(&session).is_empty())
    };
    view().child(when(
        move || !has_runs.get(),
        move || task_rows(store, session.clone()),
    ))
}

/// One request's runs in the transcript: several under a line that says
/// how they are doing ("3 agents working together · 1 done") and, once
/// every one ended, how they ended, folded when all of them finished; a
/// run alone is its row. Each row says what its run is doing or how it
/// ended, with Stop while it runs, and opens its steps.
fn run_group(store: Store, session: String, key: String, headed: bool) -> View {
    let runs = computed(move || {
        store
            .runs_for(&session)
            .into_iter()
            .filter(|run| run_key(run) == key)
            .collect::<Vec<_>>()
    });
    let several = computed(move || runs.with(Vec::len) > 1);
    // Folded once every one finished; open while any runs, or when one
    // did not finish. The owner's own unfolding stands until that changes.
    let settled =
        computed(move || runs.with(|runs| runs.iter().all(|run| run.phase == RunPhase::Succeeded)));
    let open = signal(!untracked(|| settled.get()));
    effect(move || {
        let settled = settled.get();
        untracked(|| open.set(!settled));
    });
    view().w_full().gap_1().child(dynamic(move || {
        let several = several.get();
        // The running turn's line heads its agents while it runs.
        let heading = several && (headed || runs.with(|r| r.iter().all(|r| !r.phase.is_running())));
        Some(untracked(move || {
            if !heading {
                return run_rows(store, runs).into_element();
            }
            // Stop all: the background runs still going, one press.
            let running = computed(move || {
                runs.with(|runs| {
                    runs.iter()
                        .filter(|run| run.phase.is_running() && run.background)
                        .map(|run| run.id.clone())
                        .collect::<Vec<_>>()
                })
            });
            view()
                .gap_1()
                .child(
                    view()
                        .flex_row()
                        .items_center()
                        .gap_2()
                        .child(view().flex_1().min_w(px(0.0)).child(ui::quiet_disclosure(
                            move || {
                                store.tick();
                                SharedString::from(runs.with(|runs| group_summary(runs)))
                            },
                            open,
                        )))
                        .child(when(
                            move || running.with(|ids| ids.len() > 1),
                            move || {
                                ui::btn("Stop all", Variant::Ghost, Size::Sm, move || {
                                    for id in untracked(|| running.get()) {
                                        store.stop_run(&id);
                                    }
                                })
                            },
                        )),
                )
                .child(ui::unfold(open, move || run_rows(store, runs)))
                .into_element()
        }))
    }))
}

/// A group's line: how many are working and done while any runs; once
/// all ended, how they ended, what they changed and how long it took
/// ("3 agents finished · changed 3 files · 1m 12s").
fn group_summary(runs: &[RunSummary]) -> String {
    let total = runs.len();
    let ended = runs.iter().filter(|r| !r.phase.is_running()).count();
    if ended < total {
        return group_line(total, ended);
    }
    let count = |phase: RunPhase, by_owner: bool| {
        runs.iter()
            .filter(|r| r.phase == phase && r.stopped_by_owner == by_owner)
            .count()
    };
    let finished = count(RunPhase::Succeeded, false);
    let mut line = if finished == total {
        format!("{total} agents finished")
    } else {
        let mut how = Vec::new();
        for (n, words) in [
            (finished, "finished"),
            (count(RunPhase::Failed, false), "failed"),
            (count(RunPhase::Stopped, true), "stopped by you"),
            (count(RunPhase::Stopped, false), "stopped"),
        ] {
            if n > 0 {
                how.push(format!("{n} {words}"));
            }
        }
        format!("{total} agents · {}", how.join(", "))
    };
    let mut files: Vec<&str> = runs
        .iter()
        .flat_map(|r| r.edited.iter().map(String::as_str))
        .collect();
    files.sort_unstable();
    files.dedup();
    match files.as_slice() {
        [] => {}
        [one] => line.push_str(&format!(" · changed {}", file_name(one))),
        many => line.push_str(&format!(" · changed {} files", many.len())),
    }
    let start = runs.iter().map(|r| r.started_at_ms).min().unwrap_or(0);
    let end = runs
        .iter()
        .filter_map(|r| r.ended_at_ms)
        .max()
        .unwrap_or(start);
    line.push_str(&format!(
        " · {}",
        ui::time::span(end.saturating_sub(start) / 1000)
    ));
    line
}

/// A row of the running work: a group's heading, or one run.
#[derive(Clone, PartialEq)]
enum WorkRow {
    /// Runs one request started together (`RunSummary.group`), when there
    /// are two or more: how many, and how many ended.
    Group {
        key: String,
        total: usize,
        ended: usize,
    },
    /// A run; `grouped`: it sits under its group's heading.
    Run { run: Box<RunSummary>, grouped: bool },
}

impl WorkRow {
    fn key(&self) -> String {
        match self {
            Self::Group { key, .. } => format!("group:{key}"),
            Self::Run { run, .. } => run.id.clone(),
        }
    }
}

/// The runs in order, each group of two or more under its heading.
fn work_rows(runs: &[RunSummary]) -> Vec<WorkRow> {
    let mut rows = Vec::new();
    let mut headed = std::collections::HashSet::new();
    for run in runs {
        if let Some(group) = run.group.as_deref()
            && headed.insert(group.to_string())
        {
            let members: Vec<_> = runs
                .iter()
                .filter(|other| other.group.as_deref() == Some(group))
                .collect();
            if members.len() > 1 {
                rows.push(WorkRow::Group {
                    key: group.to_string(),
                    total: members.len(),
                    ended: members.iter().filter(|m| !m.phase.is_running()).count(),
                });
                rows.extend(members.into_iter().map(|run| WorkRow::Run {
                    run: Box::new(run.clone()),
                    grouped: true,
                }));
                continue;
            }
        }
        if !rows
            .iter()
            .any(|row| matches!(row, WorkRow::Run { run: r, .. } if r.id == run.id))
        {
            rows.push(WorkRow::Run {
                run: Box::new(run.clone()),
                grouped: false,
            });
        }
    }
    rows
}

/// The running turn's line while agents it started run: "Working with
/// Codex", "Working with 3 agents · 1 done".
fn working_with(work: &crate::store::AgentWork) -> String {
    match (work.total, work.done) {
        (1, _) => format!(
            "Working with {}",
            work.kinds
                .first()
                .map(|(name, _)| name.as_str())
                .unwrap_or("an agent")
        ),
        (total, 0) => format!("Working with {total} agents"),
        (total, done) => format!("Working with {total} agents · {done} done"),
    }
}

/// A group's heading: "2 agents working together · 1 done".
fn group_line(total: usize, ended: usize) -> String {
    if ended == total {
        format!("{total} agents · all done")
    } else if ended == 0 {
        format!("{total} agents working together")
    } else {
        format!("{total} agents working together · {ended} done")
    }
}

/// A run's line: the agent (or its name), then what it is doing or how it
/// ended. Every word comes from the run's record.
fn run_line(r: &RunSummary) -> String {
    let state = match r.phase {
        RunPhase::Running if r.step.is_empty() => String::new(),
        RunPhase::Running => r.step.clone(),
        RunPhase::Succeeded => match r.edited.len() {
            0 => "Finished".into(),
            1 => format!("Finished · changed {}", file_name(&r.edited[0])),
            n => format!("Finished · changed {n} files"),
        },
        RunPhase::Failed => match &r.reason {
            Some(why) => format!("Failed: {why}"),
            None => "Failed".into(),
        },
        RunPhase::Stopped if r.stopped_by_owner => "Stopped by you".into(),
        RunPhase::Stopped => match &r.reason {
            Some(why) => format!("Stopped: {why}"),
            None => "Stopped".into(),
        },
        RunPhase::Unknown => String::new(),
    };
    // A coding agent's run names the agent, then what it is doing (its
    // task's title until its first step).
    let name = r.agent.clone().unwrap_or_else(|| r.label.clone());
    let state = if state.is_empty() && r.agent.is_some() {
        r.label.clone()
    } else {
        state
    };
    if state.is_empty() {
        name
    } else {
        format!("{name} · {state}")
    }
}

/// A path's last part.
fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// A request's runs, a row each (under its group's line when there are
/// several): its state, name and step, its time, and Stop while it runs.
/// A run's row opens its view: what it did, step by step.
fn run_rows(store: Store, runs: Memo<Vec<RunSummary>>) -> View {
    let rows = computed(move || {
        runs.with(|runs| {
            work_rows(runs)
                .into_iter()
                .filter(|row| matches!(row, WorkRow::Run { .. }))
                .collect::<Vec<_>>()
        })
    });
    view()
        .w_full()
        .gap_1()
        .child(each(rows, WorkRow::key, move |row, _| {
            match row.get_untracked() {
                WorkRow::Run { run, grouped } => run_row(store, run.id, row, grouped),
                WorkRow::Group { .. } => view(),
            }
        }))
}

fn run_row(store: Store, id: String, row: ReadSignal<WorkRow>, grouped: bool) -> View {
    let run = computed(move || match row.get() {
        WorkRow::Run { run, .. } => Some(*run),
        WorkRow::Group { .. } => None,
    });
    // A run its turn waits on is stopped by the turn's own Stop (the
    // composer's); only work that outlives its turn has its own.
    let stoppable = computed(move || {
        run.with(|r| {
            r.as_ref()
                .is_some_and(|r| r.phase.is_running() && r.background)
        })
    });
    // The runs that go on from it: the owner's follow-ups to its agent.
    let follow_ups = {
        let id = id.clone();
        computed(move || {
            store
                .all_runs()
                .iter()
                .filter(|later| later.continues.as_deref() == Some(id.as_str()))
                .count()
        })
    };
    let line = computed(move || {
        let mut words = run.with(|r| {
            r.as_ref()
                .map(|r| {
                    let line = run_line(r);
                    if r.continues.is_some() {
                        format!("{line} · follow-up")
                    } else {
                        line
                    }
                })
                .unwrap_or_default()
        });
        match follow_ups.get() {
            0 => {}
            1 => words.push_str(" · 1 follow-up"),
            n => words.push_str(&format!(" · {n} follow-ups")),
        }
        words
    });
    // How it ended by its mark as well as its colour: a check, a warning,
    // or the work mark while it runs.
    let mark = computed(move || {
        run.with(|r| match r.as_ref().map(|r| r.phase) {
            Some(RunPhase::Succeeded) => (Glyph::Check, colors::SUCCESS),
            Some(RunPhase::Failed) => (Glyph::Warning, colors::DANGER),
            Some(RunPhase::Stopped) => (Glyph::Warning, colors::WARNING),
            _ => (Glyph::Work, colors::MUTED),
        })
    });
    let open = id.clone();
    view()
        .w_full()
        .flex_row()
        .items_center()
        .gap_2()
        // A run of a group sits under its heading.
        .pl(px(if grouped { 12.0 } else { 4.0 }))
        .child(
            button()
                .flex_1()
                .min_w(px(0.0))
                .accessibility_label(move || {
                    Some(SharedString::from(format!(
                        "{}, show its steps",
                        line.get()
                    )))
                })
                .on_press(move || store.open_run(&open))
                .child(
                    view()
                        .flex_row()
                        .items_center()
                        .gap_2()
                        .child(dynamic(move || {
                            let (glyph, color) = mark.get();
                            Some(untracked(move || ui::icon(glyph).text_color(color)))
                        }))
                        .child(
                            text(move || line.get())
                                .flex_1()
                                .min_w(px(0.0))
                                .number_of_lines(1)
                                .text_size(px(ui::fit::small()))
                                .text_color(colors::MUTED),
                        ),
                ),
        )
        // How long it has run (a clock tick a second), or took once it
        // ended: a short span, from its start and end times.
        .child(
            text(move || {
                store.tick();
                run.with(|r| {
                    r.as_ref()
                        .map(|r| {
                            let end = r.ended_at_ms.unwrap_or_else(crate::store::now_ms);
                            ui::time::span(end.saturating_sub(r.started_at_ms) / 1000)
                        })
                        .unwrap_or_default()
                })
            })
            .number_of_lines(1)
            .tabular_nums()
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT),
        )
        .child(when(
            move || stoppable.get(),
            move || {
                let id = id.clone();
                ui::btn("Stop", Variant::Ghost, Size::Sm, move || {
                    store.stop_run(&id)
                })
            },
        ))
}

/// An older daemon's background work: its task list.
fn task_rows(store: Store, session: String) -> View {
    // While a turn runs, its own Stop stops everything it started (the
    // daemon lists the turn itself as a task too): one control at a time.
    // This is the work that outlives a turn.
    let tasks = computed(move || {
        if store.is_running(&session) {
            return Vec::new();
        }
        store
            .running()
            .into_iter()
            .filter(|t| t.session_id == session)
            .collect::<Vec<_>>()
    });
    view().items_center().px_4().child(each(
        tasks,
        |t| t.id.clone(),
        move |task, _| {
            let id = task.with_untracked(|t| t.id.clone());
            let started = task.with_untracked(|t| t.started_at_ms);
            view()
                .w_full()
                .max_w(px(size::COLUMN))
                .flex_row()
                .items_center()
                .gap_2()
                .pl_1()
                .child(ui::icon(Glyph::Work).text_color(colors::ACCENT))
                .child(
                    text(move || task.with(|t| t.description.clone()))
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .text_size(px(ui::fit::small()))
                        .text_color(colors::MUTED),
                )
                // How long it has run: a clock tick a second.
                .child(
                    text(move || {
                        store.tick();
                        ui::time::span(crate::store::now_ms().saturating_sub(started) / 1000)
                    })
                    .tabular_nums()
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
                )
                .child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                    store.cancel(&id)
                }))
        },
    ))
}

/// In a group conversation, instead of the composer until the owner
/// chooses: what is sent from here is posted for everyone in it.
fn group_gate(store: Store, session: String) -> View {
    let bot = untracked(|| {
        crate::store::channel_name(&store.channel(&session).unwrap_or_default(), &session)
    });
    view().items_center().px_4().pb_4().pt_2().child(
        ui::surface()
            .w_full()
            .max_w(px(size::COLUMN))
            .p_3()
            .gap_2()
            .child(
                text(format!(
                    "Other people are in this chat. A message sent from here is \
                     posted there by {bot}, for everyone in it to see."
                ))
                .text_size(px(type_scale::SM))
                .text_color(colors::MUTED),
            )
            .child(view().flex_row().gap_2().child(ui::btn(
                "Post in this group",
                Variant::Secondary,
                ui::Size::Sm,
                move || store.allow_posting(&session),
            ))),
    )
}

/// What the daemon offers in this conversation: features the agent asked
/// for (a card that turns one on), and prompts to press under its reply.
fn offers(store: Store, session: Option<String>) -> View {
    let Some(session) = session else {
        return view();
    };
    let for_features = session.clone();
    let for_prompts = session.clone();
    view()
        .items_center()
        .px_4()
        .child(dynamic(move || {
            let requests = store.feature_requests(&for_features);
            let for_features = for_features.clone();
            (!requests.is_empty()).then(|| {
                untracked(move || {
                    let mut list = view().w_full().items_center();
                    for (feature, reason) in requests {
                        let (what, how) = feature_words(feature);
                        let enable = if feature == aidaemon_client_protocol::Feature::Voice {
                            "Set it up"
                        } else {
                            "Turn it on"
                        };
                        list = list.child(
                            ui::banner(
                                format!("Your agent needs {what} {reason}. {how}"),
                                ui::Tone::Info,
                            )
                            .child(ui::btn(enable, Variant::Primary, ui::Size::Sm, {
                                let session = for_features.clone();
                                move || {
                                    if feature == aidaemon_client_protocol::Feature::Voice {
                                        store.ask(
                                            &session,
                                            "Set up voice-note transcription for me".into(),
                                        );
                                        store.dismiss_feature(feature);
                                    } else {
                                        store.enable_feature(feature);
                                        // macOS grants screen and input use in
                                        // System Settings.
                                        if feature
                                            == aidaemon_client_protocol::Feature::ComputerUse
                                            && cfg!(target_os = "macos")
                                        {
                                            drop(open_url(
                                                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
                                            ));
                                        }
                                    }
                                }
                            }))
                            .child(ui::btn(
                                "Not now",
                                Variant::Ghost,
                                ui::Size::Sm,
                                move || store.dismiss_feature(feature),
                            )),
                        );
                    }
                    list
                })
            })
        }))
        .child(dynamic(move || {
            if store.is_running(&for_prompts) {
                return None;
            }
            let prompts = store.suggestions(&for_prompts);
            (!prompts.is_empty()).then(|| {
                let session = for_prompts.clone();
                untracked(move || {
                    let mut row = view()
                        .w_full()
                        .max_w(px(size::COLUMN))
                        .flex_row()
                        .flex_wrap()
                        .gap_2()
                        .py_2();
                    for prompt in prompts {
                        let session = session.clone();
                        let text = prompt.clone();
                        row =
                            row.child(ui::btn(prompt, Variant::Outline, ui::Size::Sm, move || {
                                store.ask(&session, text.clone())
                            }));
                    }
                    row
                })
            })
        }))
}

/// A feature as the card names it, and what turning it on does.
fn feature_words(feature: aidaemon_client_protocol::Feature) -> (&'static str, &'static str) {
    use aidaemon_client_protocol::Feature;
    match feature {
        Feature::Browser => ("to browse websites", "Turning it on restarts the daemon."),
        Feature::ComputerUse => (
            "to see and use this computer's screen",
            "Turning it on restarts the daemon; your computer then asks to allow it.",
        ),
        Feature::Voice => (
            "to understand voice notes",
            "It needs a speech model on this computer; your agent can set it up.",
        ),
        Feature::Unknown => ("a feature", ""),
    }
}

/// What the agent's view moves say: a move offered while the owner types,
/// and a short note after one was taken.
fn notices(store: Store) -> View {
    view()
        .items_center()
        .px_4()
        .child(when(
            move || store.suggestion().is_some(),
            move || {
                ui::banner(
                    move || store.suggestion().unwrap_or_default(),
                    ui::Tone::Accent,
                )
                .child(ui::btn(
                    format!("Go  {}", key_label("mod+g")),
                    Variant::Primary,
                    ui::Size::Sm,
                    move || store.take_suggestion(),
                ))
                .child(ui::btn(
                    "Dismiss",
                    Variant::Ghost,
                    ui::Size::Sm,
                    move || store.dismiss_suggestion(),
                ))
            },
        ))
        // What the daemon refused or reported (a message it could not take).
        .child(when(
            move || store.notice().is_some(),
            move || {
                ui::banner(
                    move || store.notice().unwrap_or_default(),
                    ui::Tone::Neutral,
                )
                .child(ui::btn(
                    "Dismiss",
                    Variant::Ghost,
                    ui::Size::Sm,
                    move || store.dismiss_notice(),
                ))
            },
        ))
}

fn header(store: Store, session: Option<String>) -> View {
    let title = session.clone();
    let chat = session
        .clone()
        .filter(|id| store.source(id) != crate::store::Source::App);
    view()
        .h(px(ui::fit::bar()))
        .flex_row()
        .items_center()
        .gap_1()
        .px_2()
        .border_b_1()
        .border_color(colors::LINE)
        // Narrow windows reach the panes from here; wide ones when folded.
        .child(when(
            move || store.is_compact() || !store.sidebar_open(),
            move || {
                // A phone's list is a drawer of conversations, not a pane
                // to unfold: it gets its own picture, unlike the details'.
                let glyph = if untracked(|| store.is_compact()) {
                    Glyph::Chat
                } else {
                    Glyph::Sidebar
                };
                // Something elsewhere waits on the owner (amber) or works
                // (accent): a dot on the button, so the list need not be
                // opened to know.
                let elsewhere = move |is: fn(&Store, &str) -> bool| {
                    let here = store.selected();
                    store
                        .session_ids()
                        .iter()
                        .any(|id| Some(id) != here.as_ref() && is(&store, id))
                };
                view()
                    .relative()
                    .child(ui::icon_btn(glyph, "Conversations", Size::Md, move || {
                        store.open_sidebar()
                    }))
                    .child(when(
                        move || elsewhere(Store::needs_you) || elsewhere(Store::is_working),
                        move || {
                            ui::corner_dot(move || {
                                if elsewhere(Store::needs_you) {
                                    ui::Tone::Warning
                                } else {
                                    ui::Tone::Accent
                                }
                            })
                        },
                    ))
            },
        ))
        .child(
            view()
                .flex_1()
                .min_w(px(0.0))
                .px_2()
                // A new conversation has no title yet: the bar says what
                // it is, a chat or work on code.
                .child(match title {
                    Some(id) => text(move || store.title(&id))
                        .role(AccessibilityRole::Header)
                        .number_of_lines(1)
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::SEMIBOLD)
                        .into_element(),
                    None => view().items_center().child(new_kind(store)).into_element(),
                })
                // Where it lives (a chat's platform and bot) and the
                // project it works in, under its title.
                .child(when(
                    {
                        let chat = chat.clone();
                        // A new conversation's project is picked on the page.
                        let started = session.is_some();
                        move || started && (chat.is_some() || store.current_project().is_some())
                    },
                    move || {
                        let chat = chat.clone();
                        // A chat on Telegram, Discord or Slack: its mark, then
                        // the bot (and "group"), without the platform's name.
                        let glyph = chat
                            .as_deref()
                            .and_then(|id| untracked(|| store.channel(id)))
                            .and_then(|channel| ui::platform_glyph(&channel));
                        let mark = glyph.map(|glyph| {
                            ui::icon(glyph)
                                .text_color(colors::FAINT)
                                .accessibility_elements_hidden(true)
                        });
                        let line = text(move || {
                            let project = store
                                .current_project()
                                .and_then(|id| store.project(&id))
                                .map(|p| match p.path {
                                    Some(path) => format!(
                                        "{} · {}",
                                        p.name,
                                        super::sidebar::short_path(&path)
                                    ),
                                    None => p.name,
                                });
                            let place = chat.as_deref().map(|id| {
                                let place = store.place(id);
                                match glyph {
                                    Some(_) => place
                                        .split_once(" · ")
                                        .map_or(place.clone(), |(_, rest)| rest.to_owned()),
                                    None => place,
                                }
                            });
                            match (place, project) {
                                (Some(place), Some(project)) => format!("{place} · {project}"),
                                (Some(place), None) => place,
                                (None, project) => project.unwrap_or_default(),
                            }
                        })
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .text_size(px(type_scale::XS))
                        .text_color(colors::FAINT);
                        let mut row = view().flex_row().items_center().gap_1();
                        if let Some(mark) = mark {
                            row = row.child(mark);
                        }
                        row.child(line)
                    },
                )),
        )
        .child(when(
            {
                let s = session.clone();
                move || s.is_some() && !store.is_compact()
            },
            {
                let pin = session.clone();
                move || {
                    let pin = pin.clone();
                    let label = pin.clone();
                    let id = pin.clone().unwrap_or_default();
                    view()
                        .flex_row()
                        .child(dynamic(move || {
                            let pinned = store.is_pinned(&id);
                            let pin = pin.clone();
                            Some(untracked(move || {
                                ui::icon_btn(
                                    if pinned { Glyph::Pinned } else { Glyph::Pin },
                                    if pinned { "Unpin" } else { "Pin to the top" },
                                    Size::Md,
                                    move || {
                                        if let Some(id) = &pin {
                                            store.toggle_pin(id)
                                        }
                                    },
                                )
                            }))
                        }))
                        .child(ui::icon_btn(
                            Glyph::Copy,
                            "Copy conversation as Markdown",
                            Size::Md,
                            move || {
                                let _ = &label;
                                if let Some(text) = store.transcript_markdown() {
                                    set_clipboard(text);
                                    store.say("Copied the conversation");
                                }
                            },
                        ))
                }
            },
        ))
        .child({
            // The agent's browser in this conversation: watch it live (a
            // dot when it waits for the owner there).
            let session = session.clone();
            let screens = move || {
                session
                    .as_deref()
                    .map(|id| store.surfaces_in(id))
                    .unwrap_or_default()
            };
            view().child(dynamic(move || {
                let first = screens().into_iter().next()?;
                let waiting = first.handoff_pending;
                let desktop = first.session_id.is_none();
                let screen_name = first.title.clone();
                let id = first.id.clone();
                Some(untracked(move || {
                    let label = if waiting {
                        "The agent is waiting for you in its browser".to_owned()
                    } else if desktop {
                        format!("Watch the screen: {screen_name}")
                    } else {
                        "Watch the agent's browser".to_owned()
                    };
                    let button = view().relative().child(ui::icon_btn(
                        Glyph::External,
                        label,
                        Size::Md,
                        move || store.watch_surface(&id, waiting),
                    ));
                    if waiting {
                        button.child(ui::corner_dot(|| ui::Tone::Warning))
                    } else {
                        button
                    }
                    .into_element()
                }))
            }))
        })
        .child(when(
            move || store.is_compact(),
            move || {
                // A new conversation in the same project.
                ui::icon_btn(Glyph::Compose, "New conversation", Size::Md, move || {
                    store.new_in_project(store.current_project())
                })
            },
        ))
        .child(when(
            {
                // Only a coding conversation has details: a chat's steps
                // are under each reply.
                let started = session.is_some();
                move || {
                    started
                        && store.coding_on_screen()
                        && (store.is_compact() || !store.inspector_open())
                }
            },
            move || {
                view().child(dynamic(move || {
                    // A phone shows the changes themselves, the point of
                    // a coding conversation there: an icon like the bar's
                    // others, its count of files on its corner.
                    if store.is_compact() {
                        let files = store.changed_files().unwrap_or(0);
                        let label = match files {
                            0 => "Changes".to_owned(),
                            1 => "Changes, 1 file".to_owned(),
                            n => format!("Changes, {n} files"),
                        };
                        return Some(untracked(move || {
                            let button = view().relative().child(ui::icon_btn(
                                Glyph::Changes,
                                label,
                                Size::Md,
                                move || store.show_changes(),
                            ));
                            if files > 0 {
                                button.child(ui::corner_count(move || files))
                            } else {
                                button
                            }
                            .into_element()
                        }));
                    }
                    Some(untracked(move || {
                        ui::icon_btn(Glyph::Inspector, "Changes and steps", Size::Md, move || {
                            store.open_inspector()
                        })
                        .into_element()
                    }))
                }))
            },
        ))
}

/// What a new conversation is, in its header: a chat, or work on code in
/// a project's folder. Said before the first word, so the two never look
/// like the same thing.
fn new_kind(store: Store) -> View {
    ui::segmented(
        "What this conversation is",
        &["Chat", "Code"],
        move || usize::from(store.new_is_code()),
        move |at| {
            if (at == 1) != untracked(|| store.new_is_code()) {
                store.place_new(at == 1);
            }
        },
    )
}

/// A new conversation: what the header's switch chose. Both ways have
/// the same shape, a question over a room of one height (empty for a
/// chat; the project list for code), so switching changes the words and
/// nothing moves. By the composer on a phone, in the middle of a window.
/// The sidebar already lists past conversations.
fn welcome(store: Store) -> View {
    let code = computed(move || store.new_is_code());
    view()
        .flex_1()
        .items_center()
        .px_4()
        .py_4()
        .child(view().flex_1())
        .child(
            view()
                .w_full()
                .items_center()
                .gap_3()
                .child(
                    text(move || {
                        if code.get() {
                            "Which project is this for?"
                        } else {
                            "What can I help you with?"
                        }
                    })
                    .role(AccessibilityRole::Header)
                    .text_center()
                    .text_size(px(ui::fit::title()))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(colors::TEXT),
                )
                .child(
                    view()
                        .w_full()
                        .items_center()
                        .min_h(px(if ui::touch() { 72.0 } else { 56.0 }))
                        .child(when(move || code.get(), move || code_start(store))),
                ),
        )
        // On a phone all of it sits by the composer, under the thumb; in a
        // window it is centred.
        .child(view().style_when(!ui::touch(), |s| s.flex_1()))
}

/// Opens the sidebar's form for a new project (the drawer on a phone).
fn new_project_form(store: Store) {
    if !store.sidebar_open() {
        store.open_sidebar();
    }
    // The form is at the end of the list of projects.
    store.show_sidebar_view(crate::store::SidebarView::Projects);
    store.show_project_form(true);
}

/// Code: the project list, with a "+" for a new one, and under it the
/// chosen project's folder (or what to do). With no project yet, New
/// project in the list's place.
fn code_start(store: Store) -> View {
    let any = computed(move || !store.projects().is_empty());
    view().w_full().items_center().child(dynamic(move || {
        let any = any.get();
        Some(untracked(move || {
            if !any {
                return view()
                    .items_center()
                    .gap_2()
                    .child(ui::btn(
                        "New project",
                        Variant::Outline,
                        ui::Size::Sm,
                        move || new_project_form(store),
                    ))
                    .child(
                        text("A project is a folder on the daemon's computer")
                            .number_of_lines(1)
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                    );
            }
            // One card names the project and its folder; a press opens the
            // projects (recent first) with New project last.
            view().w_full().max_w(px(420.0)).child(ui::pick_list(
                ui::Picker {
                    title: "Choose a project",
                    items: move || {
                        super::sidebar::project_choices(store, store.current_project(), None)
                    },
                    pick: move |id| store.set_new_project(Some(id)),
                    action: Some(ui::PickAction {
                        label: "New project",
                        glyph: Glyph::Plus,
                        run: std::rc::Rc::new(move || new_project_form(store)),
                    }),
                },
                move || match store.current_project().and_then(|id| store.project(&id)) {
                    Some(p) => format!("Project: {}, change", p.name),
                    None => "Choose a project".to_owned(),
                },
                project_card(store),
                true,
            ))
        }))
    }))
}

/// What the Code page's card shows: the chosen project and its folder, or
/// that one is still to be chosen.
fn project_card(store: Store) -> View {
    let chosen = computed(move || store.current_project().and_then(|id| store.project(&id)));
    view()
        .w_full()
        .min_h(px(if ui::touch() { 60.0 } else { 52.0 }))
        .flex_row()
        .items_center()
        .gap_2()
        .px_3()
        .rounded(px(ui::radius::CONTROL))
        .border_1()
        .border_color(colors::EDGE)
        .bg(colors::SURFACE)
        .child(
            ui::icon(Glyph::Folder)
                .text_color(colors::MUTED)
                .accessibility_elements_hidden(true),
        )
        .child(
            view()
                .flex_1()
                .min_w(px(0.0))
                .child(
                    text(move || {
                        chosen.with(|p| {
                            p.as_ref()
                                .map_or_else(|| "Choose a project".to_owned(), |p| p.name.clone())
                        })
                    })
                    .number_of_lines(1)
                    .text_size(px(ui::fit::body()))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(move || {
                        if chosen.with(Option::is_some) {
                            colors::TEXT
                        } else {
                            colors::MUTED
                        }
                    }),
                )
                .child(
                    text(move || {
                        chosen.with(|p| match p {
                            Some(p) => match &p.path {
                                Some(path) => super::sidebar::short_path(path),
                                None => "No folder".to_owned(),
                            },
                            None => "The agent works in its folder".to_owned(),
                        })
                    })
                    .number_of_lines(1)
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
                ),
        )
        .child(
            ui::icon(Glyph::Down)
                .text_color(colors::FAINT)
                .accessibility_elements_hidden(true),
        )
}

/// The transcript, and a "Jump to latest" pill while the owner reads back.
fn transcript(store: Store, session: String) -> View {
    let id = session.clone();
    let all = computed(move || rows(store, &id));
    let last_seen = signal(usize::MAX);
    let first_seen = signal(0usize);
    // The first request with agents still running, and how many run: when
    // its rows are out of view, a line over the composer says so and
    // brings them back.
    let running = {
        let session = session.clone();
        computed(move || {
            let runs: Vec<RunSummary> = store
                .runs_for(&session)
                .into_iter()
                .filter(|run| run.phase.is_running() && run.background)
                .collect();
            let at = all.with(|rows| {
                rows.iter().position(|row| {
                    matches!(row, Row::Runs { key, .. } if runs.iter().any(|run| run_key(run) == *key))
                })
            })?;
            Some((at, runs.len()))
        })
    };
    let list = element_ref();
    let unfolding = ui::Unfolding::provide();
    let loading_id = session.clone();
    // Following the end. Sending a message or a reply starting brings
    // the end into view, as any chat does; other rows added follow it
    // while the reader is at the end. The list's own follow lost the end
    // when a reply started in a long conversation: the composer changing
    // height moved it to an old reply (live 2026-11-19).
    let shown_len = std::cell::Cell::new(untracked(|| all.with(Vec::len)));
    effect(move || {
        let (len, starts) = all.with(|rows| {
            let before = shown_len.get().min(rows.len());
            (
                rows.len(),
                rows[before..]
                    .iter()
                    .any(|row| matches!(row, Row::User(..) | Row::Live)),
            )
        });
        let before = shown_len.replace(len);
        if before == 0 {
            // It opened (or its history came): at its end.
            last_seen.set(usize::MAX);
            return;
        }
        let seen = untracked(|| last_seen.get());
        let at_end = seen == usize::MAX || seen + 2 >= before;
        if len > before && (starts || at_end) {
            list.scroll_to_end(false);
            last_seen.set(usize::MAX);
        }
    });
    view()
        .flex_1()
        // Opened before its history came (a phone restoring the last
        // conversation): say so instead of showing an empty page.
        .child(when(
            move || all.with(Vec::is_empty) && store.history_pending(&loading_id),
            move || {
                // The history comes once the link is up: until then, say
                // what the link is doing, as the composer does.
                view().flex_1().items_center().justify_center().child(
                    text(move || match store.felt() {
                        Felt::Online => "Loading…",
                        Felt::Connecting | Felt::NeedsPairing | Felt::NoDaemon => "Connecting…",
                        Felt::Reconnecting => "Reconnecting…",
                        Felt::Down { .. } => "Offline",
                    })
                    .text_size(px(type_scale::SM))
                    .text_color(colors::FAINT),
                )
            },
        ))
        .child(ui::refreshable(
            transcript_list(store, session, all, first_seen, last_seen, unfolding)
                .element_ref(&list)
                .flex_1(),
            move || store.pull_refreshing(),
            move || store.pull_refresh(),
        ))
        .child(
            view()
                .absolute()
                .bottom(px(12.0))
                .left(px(0.0))
                .right(px(0.0))
                .items_center()
                .gap_2()
                .child(when(
                    move || {
                        let Some((at, _)) = running.get() else {
                            return false;
                        };
                        let last = last_seen.get();
                        last != usize::MAX && (at < first_seen.get() || at > last)
                    },
                    move || {
                        let count = computed(move || running.get().map(|(_, n)| n));
                        view().child(dynamic(move || {
                            let n = count.get()?;
                            let label = if n == 1 {
                                "1 agent running ↑".to_string()
                            } else {
                                format!("{n} agents running ↑")
                            };
                            Some(untracked(move || {
                                ui::btn(label, Variant::Secondary, Size::Sm, move || {
                                    if let Some((at, _)) = untracked(|| running.get()) {
                                        list.scroll_to_index(
                                            u32::try_from(at).unwrap_or(0),
                                            Percentage(0.3),
                                            px(0.0),
                                            true,
                                        );
                                    }
                                })
                                .shadow_md()
                            }))
                        }))
                    },
                ))
                .child(when(
                    move || {
                        let len = all.with(Vec::len);
                        let seen = last_seen.get();
                        seen != usize::MAX && len > 0 && seen + 2 < len
                    },
                    move || {
                        ui::btn(
                            "Jump to latest ↓",
                            Variant::Secondary,
                            Size::Sm,
                            move || list.scroll_to_end(true),
                        )
                        .shadow_md()
                    },
                )),
        )
}

fn transcript_list(
    store: Store,
    session: String,
    all: Memo<Vec<Row>>,
    first_seen: Signal<usize>,
    last_seen: Signal<usize>,
    unfolding: ui::Unfolding,
) -> VirtualList {
    let count = all.with_untracked(Vec::len);
    virtual_list(
        move || all.with(Vec::len),
        move |at| {
            let row = computed(move || all.with(|r| r.get(at.index()).cloned()));
            let kind = computed(move || {
                row.with(|r| {
                    r.as_ref().map(|r| match r {
                        Row::User(..) => 0,
                        Row::Assistant(_, _, _, true, ..) => 9,
                        Row::Assistant(..) => 1,
                        Row::Note(_) => 2,
                        Row::Live => 3,
                        Row::Approval(_) => 4,
                        Row::Day(_) => 5,
                        Row::Activity(..) => 6,
                        Row::Retry => 7,
                        Row::Background => 8,
                        Row::Runs { .. } => 10,
                    })
                })
            });
            let session = session.clone();
            view().items_center().child(dynamic(move || {
                let kind = kind.get()?;
                let session = session.clone();
                Some(untracked(move || {
                    let column = view().w_full().max_w(px(size::COLUMN)).px_4().py_2();
                    let content = move || {
                        row.with(|r| match r {
                            Some(
                                Row::User(t, ..)
                                | Row::Assistant(t, ..)
                                | Row::Note(t)
                                | Row::Day(t),
                            ) => t.clone(),
                            _ => String::new(),
                        })
                    };
                    let at = move || {
                        row.with(|r| match r.as_ref() {
                            Some(Row::User(_, at, _) | Row::Assistant(_, at, ..)) => *at,
                            _ => 0,
                        })
                    };
                    let attached = move || {
                        row.with(|r| match r {
                            Some(Row::User(_, _, files) | Row::Assistant(_, _, files, ..)) => {
                                files.clone()
                            }
                            _ => Vec::new(),
                        })
                    };
                    match kind {
                        0 => column
                            .items_end()
                            .gap_1()
                            .child(files(store, attached, true))
                            .child(when(
                                move || !content().trim().is_empty(),
                                move || user_bubble(content),
                            ))
                            .child(message_footer(store, session, content, attached, at)),
                        1 => column
                            .gap_1()
                            .child(when(
                                move || !content().trim().is_empty(),
                                move || markdown(move || content().into(), || false),
                            ))
                            .child(files(store, attached, false))
                            .child(when(
                                move || {
                                    row.with(|r| {
                                        matches!(r, Some(Row::Assistant(_, _, _, _, true, _)))
                                    })
                                },
                                move || {
                                    let served = move || {
                                        row.with(|r| match r {
                                            Some(Row::Assistant(.., served)) => served.clone(),
                                            _ => None,
                                        })
                                    };
                                    reply_footer(store, content, at, served)
                                },
                            )),
                        9 => column.child(handed_off(content)),
                        2 => column.child(note(content)),
                        5 => column.child(day_break(content)),
                        3 => column.child(live(store, session)),
                        6 => column.child(activity(
                            store,
                            move || {
                                row.with(|r| match r {
                                    Some(Row::Activity(tools, secs, calls)) => {
                                        (tools.clone(), *secs, calls.clone())
                                    }
                                    _ => (Vec::new(), 0, Vec::new()),
                                })
                            },
                            false,
                        )),
                        7 => column.child(retry(store, session)),
                        8 => column.child(ui::working(
                            "Working in the background · the result comes here",
                        )),
                        10 => column.py_1().child(dynamic(move || {
                            let (key, headed) = row.with(|r| match r {
                                Some(Row::Runs { key, headed }) => Some((key.clone(), *headed)),
                                _ => None,
                            })?;
                            let session = session.clone();
                            Some(untracked(move || run_group(store, session, key, headed)))
                        })),
                        _ => column.child(dynamic(move || {
                            let id = row.with(|r| match r {
                                Some(Row::Approval(id)) => Some(id.clone()),
                                _ => None,
                            })?;
                            let approval = store.approvals().into_iter().find(|a| a.id == id)?;
                            Some(untracked(|| approvals::card(store, approval)))
                        })),
                    }
                }))
            }))
        },
    )
    .keys(move || {
        all.with(|r| {
            r.iter()
                .enumerate()
                .map(|(i, row)| row.key(i))
                .collect::<Vec<_>>()
        })
    })
    .on_viewable_items_change(move |items: &nsh_core::element::ViewableItems| {
        if let Some((first, last)) = items.viewable {
            first_seen.set(first);
            last_seen.set(last);
        }
    })
    .estimated_item_size(px(96.0))
    .initial_scroll_index(count.saturating_sub(1))
    // Following the end, except while a section unfolds: the list would
    // jump to the new end in one step under the sliding region. No
    // distance is within a negative threshold, so it keeps its place.
    .follow_end(move || px(if unfolding.active() { -1.0 } else { 48.0 }))
    .content_padding_top(px(12.0))
    .content_padding_bottom(px(12.0))
}

/// What a reply's turn did: one quiet line ("Used 4 tools · 12s") that
/// opens to every tool in order, and each tool with details opens to what
/// it was given and gave back. The owner's activity mode sets whether the
/// list starts open.
fn activity(
    store: Store,
    data: impl Fn() -> (Vec<String>, u64, Vec<ToolCallInfo>) + Clone + 'static,
    live: bool,
) -> View {
    let open = signal(untracked(|| store.activity_mode()) == ActivityMode::Expanded);
    let summary = {
        let data = data.clone();
        move || {
            let (tools, secs, calls) = data();
            aidaemon_client_core::tools::summary(&tools, &calls, secs, live)
        }
    };
    view()
        .gap_1()
        // Its own words name it: what the turn did.
        .child(ui::quiet_disclosure(summary, open))
        .child(ui::unfold(open, move || tool_list(data.clone())))
}

/// Every tool of a turn, in order, each opening to what it did.
fn tool_list(data: impl Fn() -> (Vec<String>, u64, Vec<ToolCallInfo>) + 'static) -> View {
    view().pl(px(4.0)).gap(px(2.0)).child(dynamic(move || {
        let (tools, _, calls) = data();
        Some(untracked(move || {
            let mut list = view().gap(px(2.0));
            for (n, tool) in tools.iter().enumerate() {
                list = list.child(tool_call(tool, call_of(&calls, n, tool)));
            }
            list
        }))
    }))
}

/// The details recorded for the `n`th tool, when they are its and say
/// anything.
fn call_of(calls: &[ToolCallInfo], n: usize, tool: &str) -> Option<ToolCallInfo> {
    calls
        .get(n)
        .filter(|c| c.name == tool && !(c.input.is_empty() && c.output.is_empty()))
        .cloned()
}

/// One tool of the list: what it did to what, and once opened what it
/// was given and gave back.
fn tool_call(tool: &str, call: Option<ToolCallInfo>) -> View {
    let Some(call) = call else {
        // An older daemon's bare name: in words, nothing to open.
        let step = aidaemon_client_core::tools::step(&ToolCallInfo {
            name: tool.to_owned(),
            ..ToolCallInfo::default()
        });
        return ui::tool_row(step.verb, step.subject, None);
    };
    let step = aidaemon_client_core::tools::step(&call);
    let open = signal(false);
    let input = step.more_input.then(|| call.input.clone());
    let command = step.command_output;
    view()
        .child(ui::tool_row(step.verb, step.subject, Some(open)))
        .child(ui::unfold(open, move || {
            ui::call_detail(input.clone(), call.output.clone(), command)
        }))
}

/// A reply that only says the work went on in the background: one quiet
/// line, not a message to read (its report is).
fn handed_off(content: impl Fn() -> String + 'static) -> View {
    view()
        .flex_row()
        .items_start()
        .gap_2()
        .child(
            ui::icon(Glyph::Work)
                .text_color(colors::FAINT)
                .accessibility_elements_hidden(true),
        )
        .child(
            text(content)
                .flex_1()
                .min_w(px(0.0))
                .text_size(px(ui::fit::small()))
                .text_color(colors::MUTED),
        )
}

/// Under a failed or stopped turn: send its message again.
fn retry(store: Store, session: String) -> View {
    view().flex_row().child(ui::btn_icon(
        Glyph::Retry,
        "Retry",
        Variant::Outline,
        Size::Sm,
        move || store.retry(&session),
    ))
}

/// When a message was said, quietly: "3:42 PM".
fn stamp(at: impl Fn() -> u64 + 'static) -> Text {
    text(move || {
        let at = at();
        if at == 0 {
            String::new()
        } else {
            ui::time::clock(at)
        }
    })
    .tabular_nums()
    .text_size(px(type_scale::XS))
    .text_color(colors::FAINT)
}

/// Under a reply: the provider's mark and the model that wrote it when the
/// daemon said, then its time and Copy.
fn reply_footer(
    store: Store,
    content: impl Fn() -> String + Clone + 'static,
    at: impl Fn() -> u64 + 'static,
    served: impl Fn() -> Option<ServedBy> + Clone + 'static,
) -> View {
    // What it is (model · time) and what can be done with it (Copy) are
    // two groups: a dot inside the first, a wider gap between them.
    let about = view()
        .flex_row()
        .items_center()
        .gap_1()
        .flex_shrink(1.0)
        .min_w(px(0.0))
        .child(dynamic(move || {
            let served = served()?;
            Some(untracked(move || served_by(served)))
        }))
        .child(stamp(at).flex_shrink(0.0));
    view()
        .flex_row()
        .items_center()
        .gap_4()
        .min_w(px(0.0))
        .child(about)
        // An icon, as chat apps keep it: the reply is what is read.
        .child(ui::icon_btn(Glyph::Copy, "Copy", Size::Sm, {
            let content = content.clone();
            move || {
                set_clipboard(content());
                store.say("Copied the reply");
            }
        }))
        .child(read_aloud(store, content))
}

/// Beside Copy: read the reply aloud, and Stop while it reads. Shown only
/// where something can speak (the browser's voice, or macOS's).
fn read_aloud(store: Store, content: impl Fn() -> String + 'static) -> View {
    if !crate::link::can_speak() {
        return view();
    }
    view().child(dynamic(move || {
        let text = content();
        let key = reply_key(&text);
        let on = store.speaking() == Some(key);
        Some(untracked(move || {
            let press = move || store.read_aloud(key, &text);
            if on {
                ui::icon_btn(Glyph::Stop, "Stop reading", Size::Sm, press)
                    .text_color(colors::ACCENT)
            } else {
                ui::icon_btn(Glyph::Speaker, "Read aloud", Size::Sm, press)
            }
        }))
    }))
}

/// A reply is known by its words: the key the store's one reading holds.
fn reply_key(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

/// Which model wrote a reply: the provider's mark when it has one, then
/// the model, cut short on a narrow screen, and the dot before the time.
fn served_by(served: ServedBy) -> View {
    let mut row = view()
        .flex_row()
        .items_center()
        .gap_1()
        .flex_shrink(1.0)
        .min_w(px(0.0));
    if let Some(glyph) = ui::provider_glyph(&served.preset) {
        row = row.child(ui::icon_inline(glyph).accessibility_elements_hidden(true));
    }
    row.child(
        text(served.model)
            .truncate()
            .min_w(px(0.0))
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT),
    )
    // Kept when the model is cut short, so the time never reads as part
    // of its name.
    .child(
        text("·")
            .flex_shrink(0.0)
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT)
            .accessibility_elements_hidden(true),
    )
}

/// Under a message of the owner's: its time, then Copy, Edit (back into
/// the composer, with its files) and Send again. Every message has them,
/// however old, and none changes the conversation above: what is sent
/// goes at the end.
fn message_footer(
    store: Store,
    session: String,
    content: impl Fn() -> String + Clone + 'static,
    attached: impl Fn() -> Vec<AttachmentRef> + Clone + 'static,
    at: impl Fn() -> u64 + 'static,
) -> View {
    let (copied, edited, edited_files) = (content.clone(), content.clone(), attached.clone());
    let running = session.clone();
    view()
        .flex_row()
        .items_center()
        .gap_1()
        .child(stamp(at))
        .child(ui::icon_btn(Glyph::Copy, "Copy", Size::Sm, move || {
            set_clipboard(copied());
            store.say("Copied the message");
        }))
        .child(ui::icon_btn(
            Glyph::Pen,
            "Edit and send again",
            Size::Sm,
            move || store.edit_again(edited(), edited_files()),
        ))
        // While the conversation works, a message is placed from the
        // composer (into the work, or after it): Edit takes it there.
        .child(when(
            move || !store.is_running(&running),
            move || {
                let (session, content, attached) =
                    (session.clone(), content.clone(), attached.clone());
                ui::icon_btn(Glyph::Retry, "Send again", Size::Sm, move || {
                    store.resend(&session, &content(), attached());
                })
            },
        ))
}

/// "Today", centred between hairlines.
fn day_break(label: impl Fn() -> String + 'static) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_3()
        .py_1()
        .child(view().flex_1().h(px(1.0)).bg(colors::LINE))
        .child(
            text(label)
                .role(AccessibilityRole::Header)
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::MEDIUM)
                .text_color(colors::FAINT),
        )
        .child(view().flex_1().h(px(1.0)).bg(colors::LINE))
}

/// The files sent with a message: pictures in place, other files as chips
/// that open them. `mine`: the owner's (shown at the right).
fn files(store: Store, attached: impl Fn() -> Vec<AttachmentRef> + 'static, mine: bool) -> View {
    view().child(dynamic(move || {
        let files = attached();
        if files.is_empty() {
            return None;
        }
        Some(untracked(move || {
            let mut list = view().flex_row().flex_wrap().gap_2();
            if mine {
                list = list.justify_end();
            }
            for file in files {
                let Some(url) = store.file_url(&file) else {
                    list = list.child(ui::file_chip(file.name.clone(), "not available"));
                    continue;
                };
                let open = url.clone();
                if file.is_image() && idiom() != Idiom::Terminal {
                    let pixels = file.width.zip(file.height);
                    let name = file.name.clone();
                    let id = file.id.clone();
                    list = list.child(ui::picture(
                        url,
                        format!("Picture: {}", file.name),
                        pixels,
                        move || {
                            store.view_picture(ui::Viewed {
                                id: id.clone(),
                                url: open.clone(),
                                name: name.clone(),
                                pixels,
                            });
                        },
                    ));
                } else {
                    list = list.child(
                        button()
                            .accessibility_label(format!("Open {}", file.name))
                            .on_press({
                                let name = file.name.clone();
                                move || crate::link::open_outside(store, open.clone(), name.clone())
                            })
                            .child(ui::file_chip(file.name.clone(), "")),
                    );
                }
            }
            list
        }))
    }))
}

/// The files attached to the message being written, each with a way to
/// take it off.
fn draft_files(store: Store) -> View {
    view().child(dynamic(move || {
        let files = store.draft_files();
        if files.is_empty() {
            return None;
        }
        Some(untracked(move || {
            let mut list = view()
                .flex_row()
                .flex_wrap()
                .gap_1()
                .px(px(4.0))
                .pt(px(2.0));
            for file in files {
                let number = file.number;
                let status = match &file.state {
                    DraftFileState::Uploading => "attaching…",
                    DraftFileState::Ready(_) => "",
                    DraftFileState::Failed(_) => "failed",
                };
                list = list.child(
                    view()
                        .flex_row()
                        .items_center()
                        .child(ui::file_chip(file.name.clone(), status))
                        .child(ui::icon_btn(
                            Glyph::Close,
                            "Remove the file",
                            Size::Sm,
                            move || store.remove_draft_file(number),
                        )),
                );
            }
            list
        }))
    }))
}

fn user_bubble(content: impl Fn() -> String + 'static) -> View {
    view()
        .max_w(percentage(0.8))
        .rounded(px(radius::SURFACE))
        .bg(colors::BUBBLE)
        .px_3()
        .py_2()
        .child(
            text(content)
                .selectable(true)
                .text_size(px(ui::fit::read()))
                .line_height(px(ui::fit::read_lh())),
        )
}

fn note(content: impl Fn() -> String + 'static) -> View {
    view().flex_row().gap_2().child(
        text(content)
            .text_size(px(type_scale::SM))
            .text_color(colors::MUTED),
    )
}

/// A reply, as the agent's Markdown. HTML in it is shown as text: what the
/// model writes never becomes markup.
fn markdown(
    content: impl Fn() -> SharedString + 'static,
    streaming: impl Fn() -> bool + 'static,
) -> impl IntoElement {
    // Inline code at the prose's scale and line (see `fit::read_code`).
    provide_context(TextViewParts::new().part(TextViewPart::Code, |s| {
        s.text_size(px(ui::fit::read_code()))
            .line_height(px(ui::fit::read_lh()))
            .px(px(ui::fit::read_code_pad()))
            .pt(px(ui::fit::read_code_pad()))
    }));
    text_view()
        .markdown(move || Some(content()))
        .options(move || MarkdownOptions {
            streaming: streaming(),
            html: false,
            ..MarkdownOptions::default()
        })
        .links(TextViewLinks::Open)
        .selectable(true)
        .gap_3()
        .text_size(px(ui::fit::read()))
        .line_height(px(ui::fit::read_lh()))
        .text_color(colors::TEXT)
}

/// The turn running now: its streamed text and what it is doing.
fn live(store: Store, session: String) -> View {
    let streamed = {
        let session = session.clone();
        computed(move || store.live(&session).map(|l| l.streamed).unwrap_or_default())
    };
    let status = computed({
        let session = session.clone();
        move || {
            if store.is_stopping(&session) {
                return "Stopping…".to_string();
            }
            // Waiting on its owner is the news, not the last tool it ran.
            let waiting = store.approvals_for(&session).len();
            if waiting > 0 {
                return if waiting == 1 {
                    "Waiting for your approval".to_string()
                } else {
                    format!("Waiting for {waiting} approvals")
                };
            }
            // While agents it started run, this line heads them (their
            // rows sit under it): it says how many and how far along, not
            // the last thing one of them reported.
            if let Some(work) = store.agent_work(&session) {
                return working_with(&work);
            }
            store
                .live(&session)
                .and_then(|l| l.status)
                .unwrap_or_else(|| "Working…".into())
        }
    });
    // The tools so far: a count at the row's end that opens to each, in
    // place (what the reply keeps once done).
    let so_far = {
        let session = session.clone();
        move || {
            store
                .live(&session)
                .map(|l| (l.tools, 0, l.calls))
                .unwrap_or_default()
        }
    };
    let count = {
        let so_far = so_far.clone();
        computed(move || so_far().0.len())
    };
    let open = signal(untracked(|| store.activity_mode()) == ActivityMode::Expanded);
    view()
        .gap_2()
        .child(when(
            move || streamed.with(|s| !s.is_empty()),
            move || markdown(move || streamed.get().into(), || true),
        ))
        // One row under the reply so far: that it is alive and what it is
        // doing, then how many tools. Stop is the composer's button.
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .min_h(px(ui::fit::row() - 8.0))
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .child(working(move || status.get())),
                )
                .child(when(
                    move || count.get() > 0,
                    move || {
                        ui::count_toggle(
                            move || match count.get() {
                                1 => "1 tool".to_owned(),
                                n => format!("{n} tools"),
                            },
                            open,
                        )
                        .accessibility_label("Tools so far")
                        .flex_shrink(0.0)
                    },
                )),
        )
        .child(ui::unfold(open, move || tool_list(so_far.clone())))
}

fn composer(store: Store, session: Option<String>) -> View {
    // At least one line: shorter, the browser shows a scroll bar for the
    // rest of it (a phone's line is 24 pt).
    let height = signal(ui::fit::read_lh());
    let online = move || store.is_online();
    let can_send = move || {
        online()
            && !store.needs_project()
            && (!store.draft().trim().is_empty()
                || store
                    .draft_files()
                    .iter()
                    .any(|f| matches!(f.state, DraftFileState::Ready(_))))
    };
    let running = {
        let session = session.clone();
        computed(move || session.as_deref().is_some_and(|id| store.is_running(id)))
    };
    let ongoing = session.is_some();
    let field = element_ref();
    let dragging = signal(false);
    // Enter's steer, held a moment so ⌘↩ can make it a queue instead.
    let steer_pending = stored(false);
    let owner = Owner::current();
    let touch = matches!(
        idiom(),
        nsh_core::platform::Idiom::Phone | nsh_core::platform::Idiom::Pad
    );
    let terminal = idiom() == nsh_core::platform::Idiom::Terminal;
    // While the conversation works, Enter adds the message to the work and
    // Mod+Enter queues it for after; Up on an empty composer brings back
    // the last message sent here.
    on_key_with(
        "mod+enter",
        KeyOptions {
            when: nsh_core::keymap::When::Always,
            ..KeyOptions::default()
        },
        move |_| {
            if running.get_untracked() {
                // The field's own submit (see `on_submit`) does not send it.
                steer_pending.replace(false);
                store.submit_placed(Some(other_placement(store)));
            } else {
                store.submit();
            }
        },
    );
    on_key_with(
        "up",
        KeyOptions {
            when: nsh_core::keymap::When::Always,
            pass_through: true,
            ..KeyOptions::default()
        },
        move |_| {
            store.recall_last();
        },
    );
    // Back to the composer when a conversation is picked or a pane
    // closes: typing goes here, not to a list that is gone (a phone keeps
    // its keyboard down until the composer is tapped).
    if !touch {
        effect(move || {
            if store.composer_focus() > 0 {
                untracked(|| field.focus());
            }
        });
    }
    // A new conversation the owner started: they asked to write, so the
    // keyboard comes up on a phone too. A composer built for it takes the
    // focus as it appears; one already on screen (a new conversation
    // started from a new one) when asked. A composer going away with
    // another conversation leaves the ask to the new one.
    let raise_now = session.is_none() && untracked(|| store.keyboard_wanted());
    {
        let own = session.clone();
        // Asks from before this composer was built are not its own.
        let seen = untracked(|| store.keyboard_ask());
        effect(move || {
            if store.keyboard_ask() == seen {
                return;
            }
            untracked(|| {
                if store.selected_untracked() == own && store.take_keyboard() {
                    // Not there yet (built in this same change): its
                    // auto-focus takes it.
                    field.focus();
                }
            });
        });
    }
    // A sent message put back to change: the owner asked to write, so the
    // keyboard comes up on a phone too.
    effect(move || {
        if store.composer_edit() > 0 {
            untracked(|| field.focus());
        }
    });
    let slash = computed(move || store.slash_matches());
    // A chat's messages from here also go to the chat itself.
    let chat_note = session
        .as_deref()
        .filter(|id| untracked(|| store.source(id)) != crate::store::Source::App)
        .map(|id| untracked(|| store.place(id)));
    // Telegram, Discord or Slack: its mark leads the note, which names the
    // bot without the platform.
    let chat_glyph = session
        .as_deref()
        .and_then(|id| untracked(|| store.channel(id)))
        .and_then(|channel| ui::platform_glyph(&channel))
        .filter(|_| chat_note.is_some());
    let chat_note = chat_note.map(|place| match chat_glyph {
        Some(_) => format!(
            "Also posted in {}",
            place
                .split_once(" · ")
                .map_or(place.as_str(), |(_, rest)| rest)
        ),
        None => format!("Also posted in {place}"),
    });
    // A phone's composer sits close to the conversation and the screen's
    // foot: every point it takes is a point of reading gone.
    let (pad_top, pad_bottom) = if touch { (4.0, 8.0) } else { (8.0, 16.0) };
    view()
        .items_center()
        .px_4()
        .pb(px(pad_bottom))
        .pt(px(pad_top))
        .child(when(
            move || slash.with(|m| !m.is_empty()),
            move || slash_list(store, slash),
        ))
        .child(match session.clone() {
            Some(id) => queued_list(store, id),
            None => view(),
        })
        .child(
            ui::surface()
                // No fill of its own in dark: a box of the page's colour
                // drawn by its edge, as the field it is (a filled slab read
                // as a black block under the conversation).
                .bg(colors::COMPOSER)
                .w_full()
                .max_w(px(size::COLUMN))
                .p(px(if touch { 4.0 } else { 6.0 }))
                .gap(px(if touch { 0.0 } else { 4.0 }))
                .focus_within(|s| s.border_color(colors::ACCENT))
                // Files dragged from the desktop attach where they are let
                // go; the border says so while they are over it.
                .border_color(move || {
                    if dragging.get() {
                        colors::ACCENT
                    } else {
                        colors::EDGE
                    }
                })
                .on_file_drag_enter(move |drag: &FileDragEvent| {
                    if store.takes_files() && !drag.uris.is_empty() {
                        dragging.set(true);
                    }
                })
                .on_file_drag_leave(move || dragging.set(false))
                .on_file_drop(move |drop: &FileDropEvent| {
                    dragging.set(false);
                    if store.takes_files() {
                        attach_dropped(store, drop.uris.clone(), drop.mime_types.clone());
                    }
                })
                .child(when(
                    move || dragging.get(),
                    || {
                        text("Drop to attach")
                            .px_1()
                            .text_size(px(type_scale::XS))
                            .text_color(colors::ACCENT)
                    },
                ))
                .child(draft_files(store))
                .child(
                    input()
                        .element_ref(&field)
                        // A pasted picture (a screenshot) is attached; its
                        // text, if any, still goes in.
                        .on_paste(move |paste: &PasteEvent| {
                            if !store.takes_files() {
                                return;
                            }
                            for file in &paste.files {
                                store.attach_file(file.name.clone(), file.mime.clone(), &file.data);
                            }
                        })
                        .value(move || store.draft())
                        .on_change(move |text: &str| store.set_draft(text.to_owned()))
                        .multiline(true)
                        // Phones keep Return for new lines; Send sends.
                        .submit_behavior(if touch {
                            SubmitBehavior::Newline
                        } else {
                            SubmitBehavior::Submit
                        })
                        .on_submit(move |_: &str| {
                            if running.get_untracked() {
                                // A browser's field submits on ⌘↩ too, before
                                // the Queue shortcut hears it: wait a moment
                                // for the shortcut to claim it.
                                steer_pending.replace(true);
                                owner.run(|| {
                                    spawn(async move {
                                        timer(std::time::Duration::from_millis(40)).await;
                                        if steer_pending.replace(false) {
                                            store.submit_placed(Some(enter_placement(store)));
                                        }
                                    })
                                });
                            } else {
                                store.submit();
                            }
                        })
                        // A phone would raise its keyboard over the page,
                        // unless the owner just started this conversation.
                        .auto_focus(!touch || raise_now)
                        // A drop of a few seconds is not news: the field
                        // reads as usual, and what is sent goes on reconnect.
                        .placeholder(move || match store.felt() {
                            Felt::Online if store.needs_project() => {
                                "Choose a project first".to_owned()
                            }
                            Felt::Online => {
                                // The field says what to write here.
                                let files = store.draft_files();
                                let ask = if files.iter().any(|f| f.marked_from.is_some()) {
                                    "Say what to do with the marked part"
                                } else if !files.is_empty() {
                                    "Say what to do with it, or just send"
                                } else if running.get() {
                                    "Add to the work or queue what's next"
                                } else if ongoing {
                                    "Reply"
                                } else if store.new_is_code() {
                                    "Describe the change, fix or task"
                                } else {
                                    "Ask anything, or tell it what to do"
                                };
                                // A phone's field is narrow: the hint about
                                // commands is the list that `/` brings.
                                if touch {
                                    ask.to_owned()
                                } else {
                                    format!("{ask} (/ for commands)")
                                }
                            }
                            Felt::Connecting | Felt::NeedsPairing | Felt::NoDaemon => {
                                "Connecting to the daemon…".to_owned()
                            }
                            // Short enough for a phone's field: what is
                            // written now still sends on reconnect.
                            Felt::Reconnecting => "Reconnecting…".to_owned(),
                            Felt::Down { .. } => "Offline: sends once it's back".to_owned(),
                        })
                        .placeholder_color(|| color(colors::FAINT))
                        .caret_color(|| color(colors::ACCENT))
                        .selection_color(|| color(colors::ACCENT))
                        .on_content_size_change(move |size: &nsh_core::geometry::Size<Points>| {
                            height.set(size.height.0.clamp(ui::fit::read_lh(), 180.0));
                        })
                        // A phone's line is tall already (25 pt): less
                        // padding round it keeps an empty box short.
                        .h(move || px(height.get() + if touch { 8.0 } else { 16.0 }))
                        .px(px(8.0))
                        .py(px(if touch { 4.0 } else { 8.0 }))
                        .text_size(px(ui::fit::read()))
                        .line_height(px(ui::fit::read_lh()))
                        .text_color(colors::TEXT)
                        .label("Message"),
                )
                .child(
                    view()
                        .flex_row()
                        .items_center()
                        .gap_1()
                        .child(when(
                            move || !terminal && store.takes_files(),
                            move || {
                                let mut ways = Vec::new();
                                if untracked(|| store.takes_pictures()) {
                                    ways.push(ui::MenuAction::new("Take a picture", move || {
                                        store.take_picture()
                                    }));
                                }
                                ways.push(ui::MenuAction::new("Pictures", move || {
                                    store.pick_pictures()
                                }));
                                ways.push(ui::MenuAction::new("Files", move || store.pick_files()));
                                ui::press_menu(Glyph::Attach, "Attach", ways)
                            },
                        ))
                        .child(dictation(store, terminal))
                        // Model and effort, in the same row; a phone's row
                        // has no room for them beside the working hint.
                        .child(when(move || !(touch && running.get()), {
                            let session = session.clone();
                            move || super::model_picker::model_picker(store, session.clone())
                        }))
                        .child(when(
                            move || chat_glyph.is_some() && !running.get() && !terminal,
                            move || {
                                ui::icon(chat_glyph.unwrap_or(Glyph::Chat))
                                    .text_color(colors::FAINT)
                                    .role(AccessibilityRole::Image)
                                    .label(chat_glyph.map_or("", |g| g.platform_name()))
                            },
                        ))
                        .child(
                            text(move || match (running.get(), touch) {
                                // A phone's buttons say it: Stop while the
                                // box is empty; Send and Queue once typed.
                                (true, true) => String::new(),
                                (true, false) => {
                                    let (enter, other) = if store.enter_queues() {
                                        ("after", "now")
                                    } else {
                                        ("now", "after")
                                    };
                                    let stops = if terminal {
                                        format!("Esc or {} stops", key_label("mod+c"))
                                    } else {
                                        "Esc stops".to_owned()
                                    };
                                    format!(
                                        "Enter sends {enter} · {} {other} · {stops}",
                                        key_label("mod+enter")
                                    )
                                }
                                (false, true) => chat_note.clone().unwrap_or_default(),
                                // Terminals rarely send Shift+Enter; Alt+Enter
                                // works in all of them.
                                (false, false) if terminal => {
                                    format!(
                                        "Enter to send · {} for a new line",
                                        key_label("alt+enter")
                                    )
                                }
                                (false, false) => match &chat_note {
                                    Some(note) => format!("{note} · Enter to send"),
                                    None => "Enter to send · Shift+Enter for a new line".to_owned(),
                                },
                            })
                            .flex_1()
                            .px(px(8.0))
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                        )
                        .child(when(
                            // Once there is something to send.
                            move || running.get() && can_send(),
                            move || {
                                // The other way: Queue, or Now when Enter queues.
                                view().child(dynamic(move || {
                                    let queues = store.enter_queues();
                                    Some(untracked(move || {
                                        let (label, name) = if queues {
                                            ("Now", "Now: the running work reads it")
                                        } else {
                                            ("Queue", "Queue: send after the running work")
                                        };
                                        ui::btn(label, Variant::Ghost, ui::Size::Sm, move || {
                                            store.submit_placed(Some(other_placement(store)))
                                        })
                                        .disabled(move || !can_send())
                                        .accessibility_label(name)
                                    }))
                                }))
                            },
                        ))
                        // While it works and the box is empty, the round
                        // button stops it (where Send was: where the thumb
                        // is); typing turns it back into Send.
                        .child(
                            when(
                                {
                                    let session = session.clone();
                                    move || {
                                        running.get()
                                            && !can_send()
                                            && session
                                                .as_deref()
                                                .is_some_and(|id| !store.is_stopping(id))
                                    }
                                },
                                {
                                    let session = session.clone();
                                    move || {
                                        let id = session.clone().unwrap_or_default();
                                        let stop = move || {
                                            haptic(nsh_core::Haptic::Impact(
                                                nsh_core::HapticImpact::Light,
                                            ));
                                            store.stop_from_composer(&id)
                                        };
                                        if terminal {
                                            ui::btn_icon(
                                                Glyph::Stop,
                                                "Stop",
                                                Variant::Primary,
                                                ui::Size::Sm,
                                                stop,
                                            )
                                        } else {
                                            ui::icon_btn_primary(Glyph::Stop, "Stop", stop)
                                        }
                                    }
                                },
                            )
                            .otherwise(move || {
                                {
                                    let send = move || {
                                        // A phone answers the press (nothing elsewhere).
                                        haptic(nsh_core::Haptic::Impact(
                                            nsh_core::HapticImpact::Light,
                                        ));
                                        if running.get_untracked() {
                                            store.submit_placed(Some(enter_placement(store)));
                                        } else {
                                            store.submit();
                                        }
                                    };
                                    // The arrow alone says Send; a terminal's
                                    // glyph needs the word.
                                    if terminal {
                                        ui::btn_icon(
                                            Glyph::Send,
                                            "Send",
                                            Variant::Primary,
                                            ui::Size::Sm,
                                            send,
                                        )
                                    } else {
                                        ui::icon_btn_primary(Glyph::Send, "Send", send)
                                    }
                                }
                                .disabled(move || !can_send())
                                .accessibility_label(move || {
                                    Some(SharedString::from(
                                        if running.get() && store.enter_queues() {
                                            "Send after the running work"
                                        } else if running.get() {
                                            "Send now: the running work reads it"
                                        } else {
                                            "Send"
                                        },
                                    ))
                                })
                            }),
                        ),
                ),
        )
}

/// Where Enter (and Send) puts a message while the agent works.
fn enter_placement(store: Store) -> MidRun {
    if store.enter_queues() {
        MidRun::Queue
    } else {
        MidRun::Steer
    }
}

/// Where ⌘↩ (and the second button) puts it: the other way.
fn other_placement(store: Store) -> MidRun {
    if store.enter_queues() {
        MidRun::Steer
    } else {
        MidRun::Queue
    }
}

/// Messages sent to wait for the running work, above the composer: each
/// can be edited in place or taken out until the agent takes it.
fn queued_list(store: Store, session: String) -> View {
    let items = {
        let session = session.clone();
        computed(move || store.queued(&session))
    };
    let editing = signal(None::<u64>);
    let draft = signal(String::new());
    view()
        .w_full()
        .max_w(px(size::COLUMN))
        .gap_1()
        .pb_1()
        .child(when(
            move || !items.with(Vec::is_empty),
            || ui::section_label("Waiting to send"),
        ))
        .child(each(
            items,
            |m| m.id,
            move |message, _| {
                let id = message.with_untracked(|m| m.id);
                let (save_session, remove_session) = (session.clone(), session.clone());
                view().child(
                    when(
                        move || editing.get() == Some(id),
                        move || {
                            let save_session = save_session.clone();
                            view()
                                .gap_1()
                                .child(ui::area("What to send", draft).label("Waiting message"))
                                .child(
                                    view()
                                        .flex_row()
                                        .gap_1()
                                        .child(ui::btn(
                                            "Save",
                                            Variant::Primary,
                                            ui::Size::Sm,
                                            move || {
                                                editing.set(None);
                                                store.edit_queued(
                                                    &save_session,
                                                    id,
                                                    Some(draft.get_untracked()),
                                                );
                                            },
                                        ))
                                        .child(ui::btn(
                                            "Cancel",
                                            Variant::Ghost,
                                            ui::Size::Sm,
                                            move || editing.set(None),
                                        )),
                                )
                        },
                    )
                    .otherwise(move || {
                        let remove_session = remove_session.clone();
                        let now_session = remove_session.clone();
                        ui::surface()
                            .flex_row()
                            .items_center()
                            .gap_2()
                            .px_3()
                            .py_1()
                            .child(ui::icon(Glyph::Clock).text_color(colors::FAINT))
                            // Read by the running work, or run after it.
                            .child(dynamic(move || {
                                let after = message.with(|m| m.after);
                                Some(untracked(move || {
                                    ui::with_hint(
                                        if after {
                                            "Runs as its own turn when the work ends"
                                        } else {
                                            "The running work reads it at its next step"
                                        },
                                        ui::badge(
                                            if after { "After" } else { "Now" },
                                            if after {
                                                ui::Tone::Neutral
                                            } else {
                                                ui::Tone::Accent
                                            },
                                        ),
                                    )
                                }))
                            }))
                            .child(
                                text(move || message.with(|m| m.text.clone()))
                                    .flex_1()
                                    .min_w(px(0.0))
                                    .number_of_lines(2)
                                    .text_size(px(type_scale::SM))
                                    .text_color(colors::MUTED),
                            )
                            // A message waiting for its own turn can be given
                            // to the running work instead: read at its next
                            // step, nothing it is doing interrupted.
                            .child(when(
                                move || message.with(|m| m.after) && store.can_send_queued_now(),
                                move || {
                                    let now_session = now_session.clone();
                                    ui::with_hint(
                                        "The running work reads it at its next step",
                                        ui::btn(
                                            "Send now",
                                            Variant::Ghost,
                                            ui::Size::Sm,
                                            move || store.send_queued_now(&now_session, id),
                                        ),
                                    )
                                },
                            ))
                            .child(ui::btn("Edit", Variant::Ghost, ui::Size::Sm, move || {
                                draft.set(message.with_untracked(|m| m.text.clone()));
                                editing.set(Some(id));
                            }))
                            .child(ui::icon_btn(
                                Glyph::Close,
                                "Remove from the queue",
                                ui::Size::Sm,
                                move || store.edit_queued(&remove_session, id, None),
                            ))
                    }),
                )
            },
        ))
}

/// The slash commands completing what is typed. Tab takes the first.
fn slash_list(store: Store, matches: Memo<Vec<aidaemon_client_protocol::SlashCommand>>) -> View {
    on_key_with(
        "tab",
        KeyOptions {
            when: nsh_core::keymap::When::Always,
            ..KeyOptions::default()
        },
        move |_| {
            if let Some(first) = matches.with_untracked(|m| m.first().cloned()) {
                store.complete_slash(&first.name);
            }
        },
    );
    ui::surface()
        .w_full()
        .max_w(px(size::COLUMN))
        .mb_1()
        .p_1()
        .child(each(
            move || matches.get(),
            |c: &aidaemon_client_protocol::SlashCommand| c.name.clone(),
            move |c, index| {
                let name = c.get_untracked().name.clone();
                let description = c.get_untracked().description.clone();
                let pick = name.clone();
                ui::row(|| false, move || store.complete_slash(&pick))
                    .accessibility_label(format!("/{name}: {description}"))
                    .child(ui::code(format!("/{name}")))
                    .child(
                        text(description)
                            .flex_1()
                            .number_of_lines(1)
                            .text_size(px(type_scale::SM))
                            .text_color(colors::MUTED),
                    )
                    .child(when(move || index.get() == 0, || ui::kbd("Tab")))
            },
        ))
}

/// Attaches files dropped from the desktop: read off the engine's thread,
/// each by its name, as the paperclip's are. Directories and files that
/// cannot be read say so.
#[cfg(not(target_arch = "wasm32"))]
fn attach_dropped(store: Store, uris: Vec<String>, mimes: Vec<String>) {
    spawn(async move {
        let read = background(async move {
            uris.iter()
                .zip(mimes.iter().chain(std::iter::repeat(&String::new())))
                .map(|(uri, mime)| {
                    let path = file_path(uri);
                    let name = std::path::Path::new(&path)
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| path.clone());
                    let bytes = std::fs::metadata(&path)
                        .ok()
                        .filter(|m| m.is_file())
                        .and_then(|_| std::fs::read(&path).ok());
                    (name, mime.clone(), bytes)
                })
                .collect::<Vec<_>>()
        })
        .await
        .unwrap_or_default();
        for (name, mime, bytes) in read {
            match bytes {
                Some(bytes) => {
                    let mime = if mime.is_empty() {
                        "application/octet-stream".to_owned()
                    } else {
                        mime
                    };
                    store.attach_file(name, mime, &bytes)
                }
                None => store.say(format!(
                    "{name} could not be attached (a folder, or unreadable)"
                )),
            }
        }
    });
}

/// The browser hands no paths: nothing to read.
#[cfg(target_arch = "wasm32")]
fn attach_dropped(_store: Store, _uris: Vec<String>, _mimes: Vec<String>) {}

/// The local path of a `file:` URL, its `%XX` escapes decoded.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
fn file_path(uri: &str) -> String {
    let rest = uri.strip_prefix("file://").unwrap_or(uri);
    // `file://localhost/…` and `file:///…` both name a local path.
    let rest = rest.strip_prefix("localhost").unwrap_or(rest);
    // Windows: `/C:/…` is `C:/…`.
    let rest = match rest.as_bytes() {
        [b'/', drive, b':', ..] if drive.is_ascii_alphabetic() => &rest[1..],
        _ => rest,
    };
    let bytes = rest.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(byte) = u8::from_str_radix(&rest[i + 1..i + 3], 16)
        {
            out.push(byte);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The microphone: speaking writes into the composer, after what is
/// already there, as the platform's recognizer hears it. Pressing it again
/// stops listening (the last words are kept). Shown only where speech can
/// be recognized: the platform's recognizer, or the browser's.
fn dictation(store: Store, terminal: bool) -> View {
    use std::{
        cell::{Cell, RefCell},
        rc::Rc,
    };

    use crate::link::Listening;
    use nsh_core::speech::{SpeechEvent, SpeechRequest};

    let available = signal(false);
    if !terminal {
        spawn(async move { available.set(crate::link::can_listen().await) });
    }
    let listening = signal(false);
    let session: Stored<Option<Rc<RefCell<Listening>>>> = stored(None);
    // Set when a send takes the draft: the session ends, and what the
    // recognizer still reports is dropped, so it never refills the cleared
    // composer.
    let sent: Stored<Option<Rc<Cell<bool>>>> = stored(None);
    let seen_send = Cell::new(untracked(|| store.last_sent()));
    effect(move || {
        let at = store.last_sent();
        if at == seen_send.replace(at) {
            return;
        }
        untracked(|| {
            sent.with(|s| {
                if let Some(s) = s {
                    s.set(true)
                }
            });
            session.with(|s| {
                if let Some(s) = s {
                    s.borrow().stop()
                }
            });
            listening.set(false);
        });
    });
    // The session outlives the button pressed to start it (the button
    // becomes Stop): it runs under the composer.
    let owner = Owner::current();
    let press = move || {
        if listening.get_untracked() {
            // The recognizer finishes what it heard, then the session ends.
            session.with(|s| {
                if let Some(s) = s {
                    s.borrow().stop()
                }
            });
            return;
        }
        let before = store.draft();
        let lead = if before.trim().is_empty() {
            String::new()
        } else {
            format!("{} ", before.trim_end())
        };
        let running = Rc::new(RefCell::new(Listening::start(SpeechRequest::default())));
        let dropped = Rc::new(Cell::new(false));
        session.replace(Some(running.clone()));
        sent.replace(Some(dropped.clone()));
        listening.set(true);
        owner.run(|| {
            spawn(async move {
                let mut dictated = Dictated::new(lead);
                loop {
                    let event = std::future::poll_fn(|cx| running.borrow_mut().poll_next(cx)).await;
                    match event {
                        Some(SpeechEvent::Result { .. }) if dropped.get() => {}
                        Some(SpeechEvent::Result { text, .. }) => {
                            let draft = dictated.write(&store.draft(), &text);
                            store.set_draft(draft)
                        }
                        Some(SpeechEvent::End { error }) => {
                            if let Some(note) = error.and_then(speech_note) {
                                store.say(note);
                            }
                            break;
                        }
                        None => break,
                        Some(_) => {}
                    }
                }
                // A newer session may have started after a send ended this one.
                if session.with(|s| s.as_ref().is_some_and(|s| Rc::ptr_eq(s, &running))) {
                    session.replace(None);
                    sent.replace(None);
                    listening.set(false);
                }
            })
        });
    };
    view().child(when(
        move || available.get(),
        move || {
            view().child(dynamic(move || {
                let on = listening.get();
                Some(untracked(move || {
                    if on {
                        ui::icon_btn(Glyph::Stop, "Stop listening", Size::Md, press)
                            .text_color(colors::ACCENT)
                    } else {
                        ui::icon_btn(Glyph::Mic, "Speak instead of typing", Size::Md, press)
                    }
                }))
            }))
        },
    ))
}

/// What dictation writes into the composer: the text that was there, then
/// what has been heard. The recognizer always reports the whole transcript
/// of the session, so when the draft is no longer what dictation last wrote
/// (the message was sent, or the owner typed) the draft as it is becomes the
/// new start and only words heard after that are added: a sent message never
/// comes back, and edits are kept.
struct Dictated {
    /// The draft before the words dictation adds.
    lead: String,
    /// How much of the transcript is already in `lead` (or was sent).
    heard: String,
    /// The draft dictation last wrote, and the transcript it wrote it from.
    written: Option<String>,
    written_transcript: String,
}

impl Dictated {
    fn new(lead: String) -> Self {
        Self {
            lead,
            heard: String::new(),
            written: None,
            written_transcript: String::new(),
        }
    }

    /// The draft once `transcript` (the whole session's, so far) is heard,
    /// given the draft as it is now.
    fn write(&mut self, draft: &str, transcript: &str) -> String {
        if self.written.as_deref().is_some_and(|w| w != draft) {
            // Changed outside dictation: keep it, and start after it.
            self.lead = if draft.trim().is_empty() {
                String::new()
            } else {
                format!("{} ", draft.trim_end())
            };
            self.heard = self.written_transcript.clone();
        }
        let new = after_heard(transcript, &self.heard);
        let text = format!("{}{}", self.lead, new.trim_start());
        self.written_transcript = transcript.to_string();
        self.written = Some(text.clone());
        text
    }
}

/// The words of `transcript` after `heard`, an earlier transcript of the
/// same session: as many words are skipped as `heard` had (the recognizer
/// may still revise words it had not settled, so their text can differ).
fn after_heard(transcript: &str, heard: &str) -> String {
    let skip = heard.split_whitespace().count();
    if skip == 0 {
        return transcript.to_string();
    }
    transcript
        .split_whitespace()
        .skip(skip)
        .collect::<Vec<_>>()
        .join(" ")
}

/// What to tell the owner when listening failed; nothing when it only
/// heard nothing.
fn speech_note(error: nsh_core::speech::SpeechError) -> Option<&'static str> {
    use nsh_core::speech::SpeechError;
    match error {
        SpeechError::NoSpeech => None,
        SpeechError::NotAllowed => Some(
            "The microphone or speech recognition is not allowed for AIDaemon in the system's settings",
        ),
        SpeechError::AudioCapture => Some("No microphone could be used"),
        SpeechError::LanguageNotSupported => Some("Speech in this language is not recognized here"),
        SpeechError::Network => {
            Some("Listening stopped: speech recognition could not reach its service")
        }
        _ => Some("Listening stopped: speech recognition failed"),
    }
}

#[cfg(test)]
mod dictation_tests {
    use super::Dictated;

    #[test]
    fn words_follow_what_was_typed() {
        let mut d = Dictated::new("Note: ".into());
        assert_eq!(d.write("Note:", "buy"), "Note: buy");
        assert_eq!(d.write("Note: buy", "buy milk"), "Note: buy milk");
    }

    #[test]
    fn a_sent_message_does_not_come_back() {
        let mut d = Dictated::new(String::new());
        assert_eq!(d.write("", "commit the change"), "commit the change");
        // Sent: the composer is empty, the recognizer still has it all.
        assert_eq!(d.write("", "commit the change continue"), "continue");
        assert_eq!(
            d.write("continue", "commit the change continue please"),
            "continue please"
        );
    }

    #[test]
    fn the_last_result_after_sending_leaves_the_composer_empty() {
        let mut d = Dictated::new(String::new());
        d.write("", "hello there");
        // Stop after sending: the recognizer reports what it heard once more.
        assert_eq!(d.write("", "hello there"), "");
    }

    #[test]
    fn typing_while_listening_is_kept() {
        let mut d = Dictated::new(String::new());
        d.write("", "fix the");
        assert_eq!(d.write("fix the bug", "fix the mic"), "fix the bug mic");
    }

    #[test]
    fn a_revised_word_skips_by_words() {
        let mut d = Dictated::new(String::new());
        d.write("", "hello word");
        assert_eq!(d.write("", "hello world again"), "again");
    }
}

#[cfg(test)]
mod drop_tests {
    use super::file_path;

    #[test]
    fn file_urls_become_paths() {
        assert_eq!(file_path("file:///tmp/a%20b.png"), "/tmp/a b.png");
        assert_eq!(file_path("file://localhost/tmp/x"), "/tmp/x");
        assert_eq!(
            file_path("file:///C:/Users/synthetic/n%C3%A9.txt"),
            "C:/Users/synthetic/né.txt"
        );
        assert_eq!(file_path("file:///tmp/100%"), "/tmp/100%");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(id: &str, group: Option<&str>, started_at_ms: u64) -> RunSummary {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "session_id": "app:synthetic",
            "label": id,
            "phase": "succeeded",
            "background": true,
            "started_at_ms": started_at_ms,
            "group": group,
        }))
        .expect("a run")
    }

    fn reply(at: u64) -> Row {
        Row::Assistant(format!("reply {at}"), at, Vec::new(), false, false, None)
    }

    #[test]
    fn a_requests_runs_sit_under_the_reply_that_started_them() {
        let mut rows = vec![
            Row::User("first".into(), 100, Vec::new()),
            reply(200),
            Row::User("second".into(), 300, Vec::new()),
            reply(400),
            Row::User("third".into(), 500, Vec::new()),
        ];
        let tail = place_runs(
            &mut rows,
            &[
                run("a", Some("task-1"), 150),
                run("b", Some("task-1"), 160),
                run("c", None, 350),
                run("d", Some("task-3"), 550),
            ],
        );
        let runs = |key: &str| Row::Runs {
            key: key.into(),
            headed: true,
        };
        assert_eq!(rows[2], runs("task-1"));
        assert_eq!(rows[5], runs("c"));
        // The third request has no reply yet: its runs come at the end.
        assert_eq!(tail, vec!["task-3".to_string()]);
        assert_eq!(rows.len(), 7);
    }

    #[test]
    fn a_launch_line_over_its_runs_gives_way_to_them() {
        let launch = Row::Assistant(
            "Started in the background: x".into(),
            200,
            Vec::new(),
            true,
            false,
            None,
        );
        let runs = Row::Runs {
            key: "task-1".into(),
            headed: true,
        };
        let rows = without_launch_lines(vec![
            Row::User("go".into(), 100, Vec::new()),
            launch.clone(),
            runs.clone(),
            // A launch line whose runs are not on record stays.
            launch.clone(),
            reply(300),
        ]);
        assert_eq!(
            rows,
            vec![
                Row::User("go".into(), 100, Vec::new()),
                runs,
                launch,
                reply(300)
            ]
        );
    }
}
