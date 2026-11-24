//! A coding agent's view, opened from one of its runs: the owner's thread
//! with that agent, as a chat. Each run is one exchange: what it was asked
//! (the owner's own follow-up on the right, aidaemon's delegation on the
//! left) and what the agent answered (its reply, the files it changed, its
//! checks, where its report and its record disagree). The run the view was
//! opened on also shows its steps, plan, permissions and cost, folded,
//! read from its record (`get_run`) and read again as it moves.
//!
//! An agent that kept its session takes a follow-up at the foot: the
//! owner's message goes to the same agent, in its own session (`in_run`),
//! continuing the thread's latest run; the view moves to the run that takes
//! it as soon as that starts.

use aidaemon_client_protocol::{
    RunDetail, RunPermissionOutcome, RunPhase, RunPlanStatus, RunStepState, RunSummary,
};
use nsh_base::button;
use nsh_core::prelude::*;

use crate::store::{Read, Store};
use crate::ui::{self, Glyph, Size, Tone, colors, section_label, type_scale};

pub fn run_view(store: Store, id: String) -> View {
    // The agent's runs, oldest first: this one's thread.
    let thread = {
        let id = id.clone();
        computed(move || thread_of(&store.all_runs(), &id))
    };
    let latest = computed(move || thread.with(|runs| runs.last().cloned()));
    // Who it is: its name ("Codex 2", or the owner's "Designer"), how it
    // is headed ("Designer · Codex"), and whether it waits on the owner.
    let who = computed(move || {
        let Some(first) = thread.with(|runs| runs.first().cloned()) else {
            return (String::new(), String::new(), false);
        };
        let agent = first.agent.clone().unwrap_or_else(|| first.label.clone());
        let person = store
            .participants()
            .into_iter()
            .find(|p| p.thread == first.id);
        let (name, title) = person.as_ref().map_or_else(
            || (agent.clone(), agent.clone()),
            |p| (p.name.clone(), p.title()),
        );
        let waiting = person.is_some_and(|p| p.waiting);
        (name, title, waiting)
    });
    let ready = computed(move || matches!(store.run_view(), Read::Ready(_)));
    let scroller = element_ref();
    let opened = element_ref();
    // At the newest message, as any chat opens; on an older run the view
    // was opened on, at that one. A new message brings the end into view.
    {
        let id = id.clone();
        effect(move || {
            let count = thread.with(Vec::len);
            let at_end = latest.with(|run| run.as_ref().is_none_or(|run| run.id == id));
            untracked(|| {
                if at_end {
                    scroller.scroll_to_end(false);
                } else if count > 0 {
                    let mut options = nsh_core::RevealOptions::default();
                    options.animated = false;
                    opened.reveal(options);
                }
            });
        });
    }
    view()
        .w_full()
        .flex_1()
        .bg(colors::SIDEBAR)
        .border_l_1()
        .border_color(colors::LINE)
        .child(
            view()
                .h(px(ui::fit::bar()))
                .flex_row()
                .items_center()
                .gap_2()
                .px_3()
                .child(ui::icon_btn(Glyph::Back, "Back", Size::Md, move || {
                    store.close_run()
                }))
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .child(
                            text(move || who.with(|(_, title, _)| title.clone()))
                                .number_of_lines(1)
                                .text_size(px(ui::fit::body()))
                                .font_weight(FontWeight::SEMIBOLD),
                        )
                        .child(
                            text(move || {
                                let waiting = who.with(|(_, _, waiting)| *waiting);
                                latest.with(|run| {
                                    run.as_ref()
                                        .map(|run| state(run, waiting))
                                        .unwrap_or_default()
                                })
                            })
                            .number_of_lines(1)
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                        ),
                ),
        )
        .child(scroll().flex_1().element_ref(&scroller).child(
            view().px_3().pt_2().pb_4().gap_4().child(dynamic(move || {
                // Drawn again once the opened run's record is read: an
                // older run's ask is its record's task.
                let _ = ready.get();
                let runs = thread.get();
                let id = id.clone();
                Some(untracked(move || {
                    let mut list = view().gap_4();
                    for run in runs {
                        let this = run.id == id;
                        let exchange = exchange(store, run, this);
                        list = list.child(if this {
                            exchange.element_ref(&opened)
                        } else {
                            exchange
                        });
                    }
                    list
                }))
            })),
        ))
        .child(follow_up(store, latest, who))
}

/// The agent's state, under its name: what its latest run is doing.
fn state(run: &RunSummary, waiting: bool) -> String {
    if waiting {
        return "Waiting on you".into();
    }
    match run.phase {
        RunPhase::Running => "Working".into(),
        _ if run.can_continue => "Ready for a follow-up".into(),
        RunPhase::Succeeded => "Finished".into(),
        RunPhase::Failed => "Failed".into(),
        RunPhase::Stopped if run.stopped_by_owner => "Stopped by you".into(),
        RunPhase::Stopped => "Stopped".into(),
        RunPhase::Unknown => String::new(),
    }
}

/// One run as an exchange: what it was asked, then what the agent
/// answered; `this`: the run the view was opened on, with its steps.
fn exchange(store: Store, run: RunSummary, this: bool) -> View {
    view()
        .gap_2()
        .child(asked(store, &run, this))
        .child(answer(store, &run))
        .child(if this {
            steps(store, &run)
        } else {
            more(store, &run)
        })
}

/// What it was asked: the owner's follow-up as the owner's message, on the
/// right; aidaemon's delegation on the left, its first lines, unfolding.
fn asked(store: Store, run: &RunSummary, this: bool) -> View {
    // A record from before runs kept what they were asked: the task the
    // opened run's record has, else its label.
    let task = match untracked(|| store.run_view()) {
        Read::Ready(detail) if this && !detail.task.is_empty() => Some(detail.task),
        _ => None,
    };
    let (words, by_owner) = run.asked.as_ref().map_or_else(
        || (task.unwrap_or_else(|| run.label.clone()), false),
        |asked| (asked.text.clone(), asked.by_owner),
    );
    if by_owner {
        return view().w_full().items_end().child(
            view()
                .max_w(percentage(0.85))
                .rounded(px(ui::radius::SURFACE))
                .bg(colors::BUBBLE)
                .px_3()
                .py_2()
                .child(text(words).selectable(true).text_size(px(ui::fit::body()))),
        );
    }
    let open = signal(false);
    let long = words.chars().count() > 240 || words.lines().count() > 4;
    let short = words.clone();
    view()
        .gap_1()
        .child(section_label("aidaemon asked"))
        .child(
            when(move || open.get() || !long, {
                let words = words.clone();
                move || {
                    text(words.clone())
                        .selectable(true)
                        .text_size(px(type_scale::SM))
                        .text_color(colors::MUTED)
                }
            })
            .otherwise(move || {
                text(short.clone())
                    .number_of_lines(4)
                    .text_size(px(type_scale::SM))
                    .text_color(colors::MUTED)
            }),
        )
        .child(when(
            move || long,
            move || {
                ui::quiet_disclosure(move || if open.get() { "Less" } else { "All of it" }, open)
            },
        ))
}

/// What the agent answered: working while it runs; its reply, the files
/// its record shows changed, its checks and open questions, and each claim
/// its record doesn't bear out, once it ended.
fn answer(store: Store, run: &RunSummary) -> View {
    let mut said = view().gap_1();
    if run.phase == RunPhase::Running {
        let step = if run.step.is_empty() {
            "Working".to_string()
        } else {
            run.step.clone()
        };
        return said.child(ui::working(step));
    }
    let reply = run
        .result
        .as_ref()
        .map(|result| result.text.clone())
        .filter(|text| !text.is_empty());
    match (&reply, run.phase) {
        (Some(reply), _) => {
            // The agent's reply is Markdown, as a model's is.
            let reply = reply.clone();
            said = said.child(super::conversation::markdown(
                move || reply.clone().into(),
                || false,
            ));
        }
        (None, RunPhase::Failed | RunPhase::Stopped) => {
            let why = run.reason.clone().unwrap_or_else(|| match run.phase {
                RunPhase::Failed => "It failed.".into(),
                _ => "It was stopped.".into(),
            });
            said = said.child(line(Glyph::Warning, Tone::Danger, why, false));
        }
        _ => {
            said = said.child(note("It ended without a reply.".into()));
        }
    }
    let coding = store.is_coding(&run.session_id);
    for path in run.edited.clone() {
        said = said.child(changed_file(store, path, coding));
    }
    if let Some(result) = &run.result {
        for (glyph, tone, words) in result_facts(result) {
            said = said.child(line(glyph, tone, words, false));
        }
    }
    said
}

/// What a run's result says beyond its reply, a line each: each check it
/// reports, as its record bears it out (a check the record doesn't show
/// run, or shows failing, is one warning, not a pass and a warning), its
/// open questions, that it needs the owner, and each other claim the
/// record doesn't bear out.
pub(crate) fn result_facts(
    result: &aidaemon_client_protocol::RunResult,
) -> Vec<(Glyph, Tone, String)> {
    use aidaemon_client_protocol::RunMismatch;
    let mut facts = Vec::new();
    let checks = result
        .report
        .as_ref()
        .map(|report| report.checks.as_slice())
        .unwrap_or_default();
    let not_run = |command: &str| {
        result
            .mismatches
            .iter()
            .any(|m| matches!(m, RunMismatch::CheckNotRun { command: c } if c == command))
    };
    let failed = |command: &str| {
        result
            .mismatches
            .iter()
            .any(|m| matches!(m, RunMismatch::CheckFailed { command: c } if c == command))
    };
    for check in checks {
        let command = &check.command;
        facts.push(if failed(command) {
            (
                Glyph::Warning,
                Tone::Danger,
                format!("{command} failed (it said passed)"),
            )
        } else if not_run(command) {
            (
                Glyph::Warning,
                Tone::Warning,
                format!(
                    "{command}: said {}, but no run of it was seen",
                    if check.passed { "passed" } else { "failed" }
                ),
            )
        } else if check.passed {
            (Glyph::Check, Tone::Success, format!("{command} passed"))
        } else {
            (Glyph::Warning, Tone::Danger, format!("{command} failed"))
        });
    }
    if let Some(report) = &result.report {
        for question in &report.open_questions {
            facts.push((Glyph::Dot, Tone::Info, format!("Open: {question}")));
        }
        if report.needs_owner {
            facts.push((Glyph::Approval, Tone::Warning, "Needs you".into()));
        }
    }
    let reported = |command: &str| checks.iter().any(|check| check.command == command);
    for mismatch in &result.mismatches {
        let words = match mismatch {
            RunMismatch::ChangeNotSeen { path } => {
                format!(
                    "Says it changed {}, but no change was seen",
                    file_name(path)
                )
            }
            RunMismatch::ChangeNotClaimed { path } => {
                format!("Changed {} without saying so", file_name(path))
            }
            RunMismatch::CheckNotRun { command } | RunMismatch::CheckFailed { command }
                if reported(command) =>
            {
                continue;
            }
            RunMismatch::CheckNotRun { command } => {
                format!("Says it ran {command}, but no run was seen")
            }
            RunMismatch::CheckFailed { command } => {
                format!("Says {command} passed, but it failed")
            }
            RunMismatch::Unknown => continue,
        };
        facts.push((Glyph::Warning, Tone::Warning, words));
    }
    facts
}

/// A file a run changed; in a coding conversation it opens its diff.
fn changed_file(store: Store, path: String, coding: bool) -> View {
    let name = file_name(&path).to_string();
    let tint = if coding {
        colors::ACCENT
    } else {
        colors::MUTED
    };
    let row = view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(ui::icon(Glyph::File).text_color(tint))
        .child(
            text(name.clone())
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(1)
                .text_size(px(type_scale::SM))
                .text_color(tint),
        );
    if coding {
        view().child(
            button()
                .accessibility_label(format!("{name}, show what changed"))
                .on_press(move || store.show_change(&path))
                .child(row),
        )
    } else {
        row
    }
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// The steps of a run the view wasn't opened on: one quiet line that
/// opens them.
fn more(store: Store, run: &RunSummary) -> View {
    if run.phase == RunPhase::Running {
        return view();
    }
    let id = run.id.clone();
    view().child(
        button()
            .self_start()
            .accessibility_label("Show its steps")
            .on_press(move || store.open_run(&id))
            .child(
                text("Steps")
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
            ),
    )
}

/// The steps, plan, permissions and cost of the run the view was opened
/// on, folded, from its record; nothing when it kept none.
fn steps(store: Store, run: &RunSummary) -> View {
    let running = run.phase == RunPhase::Running;
    let open = signal(running);
    view().child(dynamic(move || {
        let Read::Ready(detail) = store.run_view() else {
            return None;
        };
        let has = !detail.steps.is_empty()
            || !detail.plan.is_empty()
            || !detail.permissions.is_empty()
            || detail.model.is_some()
            || detail.usage.is_some();
        has.then(|| {
            untracked(move || {
                let count = detail.steps.len();
                let label = if count == 0 {
                    "Details".to_string()
                } else if count == 1 {
                    "1 step".to_string()
                } else {
                    format!("{count} steps")
                };
                view()
                    .gap_1()
                    .child(ui::quiet_disclosure(label, open))
                    .child(ui::unfold(open, move || body(detail.clone())))
            })
        })
    }))
}

/// The runs of run `id`'s thread, oldest first: its first run and every
/// follow-up in the same agent session (`RunSummary::thread`, the
/// daemon's own record).
fn thread_of(runs: &[RunSummary], id: &str) -> Vec<RunSummary> {
    let key = |run: &RunSummary| run.thread.clone().unwrap_or_else(|| run.id.clone());
    let Some(thread) = runs.iter().find(|run| run.id == id).map(key) else {
        return Vec::new();
    };
    let mut thread: Vec<RunSummary> = runs
        .iter()
        .filter(|run| key(run) == thread)
        .cloned()
        .collect();
    thread.sort_by_key(|run| run.started_at_ms);
    thread
}

/// A follow-up at the view's foot, continuing the thread's latest run: only
/// when its agent kept its session, it ended, and nothing goes on from it
/// yet. Once sent, the message shows as sent until the run that takes it
/// starts, and the view moves to that run.
fn follow_up(
    store: Store,
    latest: Memo<Option<RunSummary>>,
    who: Memo<(String, String, bool)>,
) -> View {
    let draft = signal(String::new());
    // What was sent, and to which run, until a run continues it.
    let sent: Signal<Option<(String, String)>> = signal(None);
    let continued = computed(move || {
        let (_, earlier) = sent.get()?;
        store
            .all_runs()
            .into_iter()
            .filter(|run| run.continues.as_deref() == Some(earlier.as_str()))
            .max_by_key(|run| run.started_at_ms)
            .map(|run| run.id)
    });
    effect(move || {
        if let Some(next) = continued.get() {
            sent.set(None);
            untracked(|| store.open_run(&next));
        }
    });
    let open = computed(move || {
        latest.with(|run| {
            run.as_ref()
                .is_some_and(|run| run.can_continue && run.phase != RunPhase::Running)
        })
    });
    let name = computed(move || who.with(|(name, _, _)| name.clone()));
    let send = move || {
        let text = draft.get_untracked();
        if text.trim().is_empty() {
            return;
        }
        let Some((session, run)) = latest.with_untracked(|run| {
            run.as_ref()
                .map(|run| (run.session_id.clone(), run.id.clone()))
        }) else {
            return;
        };
        store.follow_up(&session, &run, &text);
        draft.set(String::new());
        sent.set(Some((text.trim().to_string(), run)));
    };
    view()
        .child(when(
            move || sent.with(Option::is_some),
            move || {
                view()
                    .px_3()
                    .py_2()
                    .gap_1()
                    .items_end()
                    .border_t_1()
                    .border_color(colors::LINE)
                    .child(
                        view()
                            .max_w(percentage(0.85))
                            .rounded(px(ui::radius::SURFACE))
                            .bg(colors::BUBBLE)
                            .px_3()
                            .py_2()
                            .child(
                                text(move || {
                                    sent.with(|sent| {
                                        sent.as_ref()
                                            .map(|(text, _)| text.clone())
                                            .unwrap_or_default()
                                    })
                                })
                                .text_size(px(ui::fit::body())),
                            ),
                    )
                    .child(
                        text(move || format!("Sending to {}…", name.get()))
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                    )
            },
        ))
        .child(when(
            move || open.get() && sent.with(Option::is_none),
            move || {
                view()
                    .px_3()
                    .py_2()
                    .border_t_1()
                    .border_color(colors::LINE)
                    .child(
                        view()
                            .flex_row()
                            .items_center()
                            .gap_2()
                            .child(
                                ui::field("Reply…", draft)
                                    .label("Follow-up for the agent")
                                    .flex_1()
                                    .min_w(px(0.0))
                                    .on_submit(move |_: &str| send()),
                            )
                            .child(ui::icon_btn_primary(Glyph::Send, "Send to the agent", send)),
                    )
            },
        ))
}

fn note(words: String) -> View {
    view().px_1().child(
        text(words)
            .text_size(px(type_scale::SM))
            .text_color(colors::FAINT),
    )
}

/// The run's steps, plan, permissions, and what it ran on and cost.
fn body(detail: RunDetail) -> View {
    let mut page = view().gap_3().pt_1().pl(px(4.0));
    if !detail.plan.is_empty() {
        let mut plan = view().gap_1();
        for item in &detail.plan {
            let (glyph, tone) = match item.status {
                RunPlanStatus::Done => (Glyph::Check, Tone::Success),
                RunPlanStatus::InProgress => (Glyph::Work, Tone::Accent),
                _ => (Glyph::Dot, Tone::Neutral),
            };
            plan = plan.child(line(glyph, tone, item.text.clone(), false));
        }
        page = page.child(section("Plan", plan));
    }
    if !detail.steps.is_empty() {
        let mut steps = view().gap_1();
        for step in &detail.steps {
            let (glyph, tone) = match step.state {
                RunStepState::Done => (Glyph::Check, Tone::Success),
                RunStepState::Failed => (Glyph::Warning, Tone::Danger),
                _ => (Glyph::Work, Tone::Accent),
            };
            steps = steps.child(line(glyph, tone, step.words.clone(), step.by_helper));
        }
        page = page.child(steps);
    }
    if !detail.permissions.is_empty() {
        let mut asked = view().gap_1();
        for permission in &detail.permissions {
            let (glyph, tone, outcome) = match permission.outcome {
                RunPermissionOutcome::Allowed => (Glyph::Check, Tone::Success, "allowed"),
                RunPermissionOutcome::Refused => (Glyph::Warning, Tone::Danger, "refused"),
                RunPermissionOutcome::Declined => (Glyph::Warning, Tone::Warning, "declined"),
                RunPermissionOutcome::Unanswered => (Glyph::Warning, Tone::Warning, "not answered"),
                RunPermissionOutcome::Unknown => (Glyph::Dot, Tone::Neutral, ""),
            };
            let mut words = format!("{} · {outcome}", permission.label);
            if let Some(why) = &permission.reason {
                words.push_str(&format!(": {why}"));
            }
            asked = asked.child(line(glyph, tone, words, false));
        }
        page = page.child(section("Permissions", asked));
    }
    let mut facts = Vec::new();
    if let Some(model) = &detail.model {
        facts.push(format!("Model: {model}"));
    }
    if let Some(usage) = &detail.usage {
        let mut tokens = format!(
            "{} tokens in, {} out",
            ui::time::compact_count(usage.input_tokens),
            ui::time::compact_count(usage.output_tokens)
        );
        if let Some(cost) = usage.cost_usd {
            tokens.push_str(&format!(" · ${cost:.2}"));
        }
        facts.push(tokens);
    }
    if let Some(dir) = &detail.working_dir {
        facts.push(format!("In {dir}"));
    }
    if let Some(session) = &detail.agent_session_id {
        facts.push(format!("Its session: {session}"));
    }
    if !facts.is_empty() {
        let mut list = view().gap_1();
        for fact in facts {
            list = list.child(
                text(fact)
                    .selectable(true)
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
            );
        }
        page = page.child(list);
    }
    page
}

fn section(title: &'static str, content: View) -> View {
    view().gap_1().child(section_label(title)).child(content)
}

/// One step, plan item or permission: its mark and its words; a helper's
/// step sits in under the agent's own.
fn line(glyph: Glyph, tone: Tone, words: String, nested: bool) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_2()
        .pl(px(if nested { 16.0 } else { 0.0 }))
        .child(ui::icon(glyph).text_color(tone.color()))
        .child(
            text(words)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(2)
                .selectable(true)
                .text_size(px(type_scale::SM)),
        )
}
