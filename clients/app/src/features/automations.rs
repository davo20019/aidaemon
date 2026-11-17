//! The Automations page: everything that runs on its own (scheduled work
//! and reminders, watching automations, health checks), next to run first,
//! each with Pause or Resume, Run now (scheduled work), Details (its card
//! and recent runs), Edit (scheduled work: name, when, what it does) or
//! Change in chat, and Cancel (asked twice). New ones are made by asking
//! the agent, so the page starts a conversation for that. Read on open,
//! every 30 seconds while open and no form is open, on Refresh, and after
//! each change (its answer brings the list).

use aidaemon_client_protocol::{
    AutomationChange, AutomationKind, AutomationRun, AutomationSummary, RunOutcome,
};
use nsh_core::prelude::*;

use crate::store::{AutomationsRead, Store, now_ms};
use crate::ui::{self, Glyph, Size, Tone, Variant, badge, colors, size, time, type_scale};

pub fn page(store: Store) -> View {
    set_interval(std::time::Duration::from_secs(30), move || {
        if store.automation_editing().is_none() {
            store.refresh_automations()
        }
    });
    view()
        .flex_1()
        .min_w(px(0.0))
        .child(header(store))
        .child(
            scroll()
                .flex_1()
                .child(view().items_center().child(dynamic(move || {
                    let read = store.automations();
                    Some(untracked(move || body(store, read)))
                }))),
        )
}

fn header(store: Store) -> View {
    view()
        .h(px(ui::fit::bar()))
        .flex_row()
        .items_center()
        .gap_1()
        .px_2()
        .border_b_1()
        .border_color(colors::LINE)
        .child(when(
            move || store.is_compact() || !store.sidebar_open(),
            move || {
                ui::icon_btn(Glyph::Sidebar, "Conversations", Size::Md, move || {
                    store.open_sidebar()
                })
            },
        ))
        .child(
            text("Automations")
                .flex_1()
                .px_2()
                .role(AccessibilityRole::Header)
                .text_size(px(ui::fit::body()))
                .font_weight(FontWeight::SEMIBOLD),
        )
        .child(ui::btn_icon(
            Glyph::Plus,
            "New",
            Variant::Ghost,
            Size::Sm,
            move || new_automation(store),
        ))
        .child(ui::icon_btn(Glyph::Retry, "Refresh", Size::Md, move || {
            store.refresh_automations()
        }))
}

/// A new automation is asked for in words: a new conversation, started.
fn new_automation(store: Store) {
    store.select(None);
    store.set_draft("Every weekday at 9am, ".to_owned());
}

fn body(store: Store, read: AutomationsRead) -> View {
    let column = view()
        .w_full()
        .max_w(px(size::COLUMN))
        .px_4()
        .py_4()
        .gap_3();
    match read {
        AutomationsRead::NotYet | AutomationsRead::Loading => {
            column.child(ui::working("Reading your automations…"))
        }
        AutomationsRead::Failed(reason) => column.child(ui::banner(reason, Tone::Danger).child(
            ui::btn("Try again", Variant::Secondary, Size::Sm, move || {
                store.refresh_automations()
            }),
        )),
        AutomationsRead::Ready(list) if list.is_empty() => column
            .child(ui::empty(
                "Nothing runs on its own yet",
                "Ask in a conversation: \"every weekday at 8, summarize my inbox\", or \
                 \"watch the status page and tell me when it changes\".",
            ))
            .child(view().flex_row().justify_center().child(ui::btn(
                "New automation",
                Variant::Primary,
                Size::Sm,
                move || new_automation(store),
            ))),
        AutomationsRead::Ready(list) => {
            let count = list.len();
            let mut column = column.child(
                text(format!(
                    "{count} automation{}, next to run first",
                    if count == 1 { "" } else { "s" }
                ))
                .px_1()
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT),
            );
            for automation in list {
                column = column.child(card(store, automation));
            }
            column
        }
    }
}

fn kind_word(kind: AutomationKind) -> &'static str {
    match kind {
        AutomationKind::Scheduled => "Scheduled",
        AutomationKind::Watching => "Watching",
        AutomationKind::Probe => "Health check",
        AutomationKind::Unknown => "Automation",
    }
}

/// Its state when it is not simply on.
fn state_badge(status: &str) -> Option<(&'static str, Tone)> {
    match status {
        "active" => None,
        "paused" => Some(("Paused", Tone::Neutral)),
        "failed" | "stalled" => Some(("Failed", Tone::Danger)),
        "held_after_failures" => Some(("Paused after failures", Tone::Danger)),
        "awaiting_input" => Some(("Waiting for you", Tone::Warning)),
        "pending_confirmation" => Some(("Not confirmed yet", Tone::Warning)),
        _ => Some(("Off", Tone::Neutral)),
    }
}

fn outcome_word(outcome: RunOutcome) -> (&'static str, Tone) {
    match outcome {
        RunOutcome::Running => ("Running", Tone::Info),
        RunOutcome::Done => ("Done", Tone::Success),
        RunOutcome::Failed => ("Failed", Tone::Danger),
        RunOutcome::Blocked => ("Blocked", Tone::Warning),
        RunOutcome::Cancelled => ("Cancelled", Tone::Neutral),
        // Due while the daemon was off: it never ran. Not a failure.
        RunOutcome::Missed => ("Missed", Tone::Warning),
        RunOutcome::Unknown => ("Ended", Tone::Neutral),
    }
}

/// "Mon 9:02 AM".
fn ran_at(at_ms: u64) -> String {
    format!(
        "{} {}",
        time::day_label(at_ms, now_ms()),
        time::clock(at_ms)
    )
}

/// Its latest run, recessed in the card: how it ended and when on one
/// line (a missed run says it was due, not that it ran), what it said
/// under it. `new`: it reported something since the last look.
fn last_run(run: &AutomationRun, new: bool) -> View {
    let (word, tone) = outcome_word(run.outcome);
    let when = if run.outcome == RunOutcome::Missed {
        format!("Last run due {}", ran_at(run.started_ms))
    } else {
        format!("Last run {}", ran_at(run.started_ms))
    };
    let mut head = view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(ui::badge_dot(word, tone))
        .child(
            text(when)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(1)
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT),
        );
    if new {
        head = head.child(badge("New since you last looked", Tone::Accent));
    }
    let mut block = ui::inset().gap_1().child(head);
    if let Some(summary) = run.summary.as_deref() {
        block = block.child(
            text(summary.to_owned())
                .number_of_lines(3)
                .text_size(px(ui::fit::small()))
                .text_color(if run.outcome == RunOutcome::Failed {
                    colors::DANGER
                } else {
                    colors::TEXT
                }),
        );
    }
    block
}

/// Its recent runs, newest first: each run's end and what it said.
fn history(runs: Vec<AutomationRun>) -> View {
    let mut list = view().gap_2().pt_1();
    for run in runs {
        let (word, tone) = outcome_word(run.outcome);
        let mut row = view().gap(px(2.0)).child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .child(ui::status(word, tone))
                .child(
                    text(ran_at(run.started_ms))
                        .text_size(px(type_scale::XS))
                        .text_color(colors::MUTED),
                ),
        );
        if let Some(summary) = run.summary {
            row = row.child(
                text(summary)
                    .number_of_lines(4)
                    .text_size(px(type_scale::SM))
                    .text_color(colors::MUTED),
            );
        }
        list = list.child(row);
    }
    list
}

fn card(store: Store, automation: AutomationSummary) -> View {
    let id = automation.id.clone();
    let title = automation
        .name
        .clone()
        .unwrap_or_else(|| automation.what.clone());
    // How often, then when next: "Every 8 hours · next Tomorrow 1:11 AM".
    let mut schedule = automation.how_often.clone();
    if let Some(next) = automation.next_run_ms {
        let now = now_ms();
        schedule.push_str(&format!(
            " · next {} {}",
            time::day_label(next, now),
            time::clock(next)
        ));
    }
    let paused = matches!(automation.status.as_str(), "paused" | "failed" | "stalled");
    let busy = {
        let id = id.clone();
        move || store.automation_busy().as_deref() == Some(id.as_str())
    };
    let confirming = signal(false);
    let mut heading = view().flex_row().items_start().gap_2().child(
        text(title)
            .flex_1()
            .min_w(px(0.0))
            .number_of_lines(2)
            .role(AccessibilityRole::Header)
            .text_size(px(ui::fit::body()))
            .font_weight(FontWeight::SEMIBOLD),
    );
    if let Some((label, tone)) = state_badge(&automation.status) {
        heading = heading.child(badge(label, tone));
    }
    // Its kind as a quiet tag before its cadence.
    let meta = view()
        .flex_row()
        .flex_wrap()
        .items_center()
        .gap_2()
        .child(badge(kind_word(automation.kind), Tone::Neutral))
        .child(
            text(schedule)
                .flex_1()
                .min_w(px(0.0))
                .text_size(px(ui::fit::small()))
                .text_color(colors::MUTED),
        );
    let mut c = ui::surface().p_4().gap_2().child(heading).child(meta);
    // A run waiting for the owner on a page (a sign-in, a code): say so on
    // its card, with the way to do it.
    if let Some(session) = automation.session_id.clone() {
        c = c.child(dynamic(move || {
            let surface = store.waiting_surface(&session)?;
            Some(untracked(move || {
                let what = match surface.site.as_deref() {
                    Some(site) => format!("Waiting for you on {site}"),
                    None => "Waiting for you in its browser".to_owned(),
                };
                let id = surface.id.clone();
                view()
                    .gap_2()
                    .child(ui::banner(what, Tone::Warning))
                    .child(view().flex_row().child(ui::btn(
                        "Open the live view",
                        Variant::Primary,
                        Size::Sm,
                        move || store.watch_surface(&id, true),
                    )))
                    .into_element()
            }))
        }));
    }
    if automation.name.is_some() {
        c = c.child(
            text(automation.what.clone())
                .number_of_lines(2)
                .text_size(px(ui::fit::small()))
                .text_color(colors::MUTED),
        );
    }
    // What it did, not only what it will do.
    match automation.recent_runs.first() {
        Some(run) => c = c.child(last_run(run, store.run_is_new(run))),
        None => {
            // A failing check keeps no runs here: its badge says it failed,
            // so "hasn't run" would contradict it.
            let failing = matches!(automation.status.as_str(), "failed" | "stalled");
            c = c.child(
                ui::inset().child(
                    text(if failing {
                        "It is failing. Its runs are not recorded here; Details has what is known."
                    } else {
                        "It hasn't run yet."
                    })
                    .text_size(px(ui::fit::small()))
                    .text_color(colors::FAINT),
                ),
            )
        }
    }
    let showing = signal(false);
    let act = {
        let id = id.clone();
        move |change: AutomationChange| {
            let id = id.clone();
            move || store.change_automation(&id, change)
        }
    };
    // Waiting on the owner (a question, a confirmation), it neither runs
    // nor pauses: the answer is in the chat it asked in.
    let waiting = matches!(
        automation.status.as_str(),
        "awaiting_input" | "pending_confirmation"
    );
    if waiting {
        c = c.child(
            text("It is waiting for your answer in the chat where it asked.")
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT),
        );
    }
    // Paused after failures: its next good run turns it back on.
    let held = automation.status == "held_after_failures";
    let failed = matches!(automation.status.as_str(), "failed" | "stalled");
    let mut actions = view().flex_row().flex_wrap().items_center().gap_1();
    if held {
        actions = actions.child(
            ui::btn(
                "Try again",
                Variant::Secondary,
                Size::Sm,
                act(AutomationChange::RunNow),
            )
            .disabled(busy.clone()),
        );
    } else if !waiting {
        actions = actions.child(
            if paused {
                ui::btn(
                    if failed { "Try again" } else { "Resume" },
                    Variant::Secondary,
                    Size::Sm,
                    act(AutomationChange::Resume),
                )
            } else {
                ui::btn(
                    "Pause",
                    Variant::Secondary,
                    Size::Sm,
                    act(AutomationChange::Pause),
                )
            }
            .disabled(busy.clone()),
        );
    }
    if automation.kind == AutomationKind::Scheduled && !held {
        actions = actions.child(
            ui::btn(
                "Run now",
                Variant::Ghost,
                Size::Sm,
                act(AutomationChange::RunNow),
            )
            .disabled(busy.clone()),
        );
    }
    // Change it: here when it can be (scheduled work), else by asking.
    let editing = {
        let id = id.clone();
        move || store.automation_editing().as_deref() == Some(id.as_str())
    };
    match automation.edit.clone() {
        Some(_) => {
            let id = id.clone();
            actions = actions.child(
                ui::btn("Edit", Variant::Ghost, Size::Sm, move || {
                    store.set_automation_editing(Some(id.clone()))
                })
                .disabled(busy.clone()),
            );
        }
        None => {
            let title = automation
                .name
                .clone()
                .unwrap_or_else(|| automation.what.clone());
            actions = actions.child(ui::btn(
                "Change in chat",
                Variant::Ghost,
                Size::Sm,
                move || store.change_automation_in_chat(&title),
            ));
        }
    }
    // Details is a chevron at the row's end, so the actions keep one row
    // on a phone.
    actions = actions.child(view().flex_1()).child(
        when(showing, move || {
            ui::icon_btn(Glyph::Up, "Hide details", Size::Sm, move || {
                showing.set(false)
            })
        })
        .otherwise(move || {
            ui::icon_btn(Glyph::Down, "Details", Size::Sm, move || showing.set(true))
        }),
    );
    // Cancelling is rare and final: it sits at the foot of Details, and
    // asks again before it happens.
    let cancel = act(AutomationChange::Cancel);
    let cancel_row = move || {
        let cancel = cancel.clone();
        view().flex_row().pt_2().child(
            when(confirming, move || {
                let cancel = cancel.clone();
                view()
                    .flex_row()
                    .flex_wrap()
                    .items_center()
                    .gap_1()
                    .child(
                        text("Cancel it for good?")
                            .pr_1()
                            .text_size(px(ui::fit::small()))
                            .text_color(colors::MUTED),
                    )
                    .child(ui::btn("Yes, cancel", Variant::Danger, Size::Sm, {
                        let cancel = cancel.clone();
                        move || {
                            confirming.set(false);
                            cancel()
                        }
                    }))
                    .child(ui::btn("Keep it", Variant::Ghost, Size::Sm, move || {
                        confirming.set(false)
                    }))
            })
            .otherwise(move || {
                ui::btn("Cancel automation", Variant::Outline, Size::Sm, move || {
                    confirming.set(true)
                })
            }),
        )
    };
    let runs = automation.recent_runs.clone();
    let lines = automation.details.clone();
    let form = automation.edit.clone().map(|edit| {
        let id = id.clone();
        let name = automation.name.clone().unwrap_or_default();
        move || edit_form(store, id.clone(), name.clone(), edit.clone())
    });
    let mut c = c.child(actions).child(when(showing, move || {
        details(lines.clone(), runs.clone()).child(cancel_row())
    }));
    if let Some(form) = form {
        c = c.child(when(editing, form));
    }
    c
}

/// All of it: what it may do, when it tells you, when it ends and the rest
/// of its card, then its recent runs.
fn details(lines: Vec<String>, runs: Vec<AutomationRun>) -> View {
    let mut list = view().gap_1().pt_1().child(ui::separator().mb_2());
    for line in lines {
        list = list.child(
            text(line)
                .text_size(px(ui::fit::small()))
                .text_color(colors::MUTED),
        );
    }
    if runs.is_empty() {
        return list;
    }
    list.child(
        text("Recent runs")
            .pt_2()
            .text_size(px(type_scale::XS))
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(colors::MUTED),
    )
    .child(history(runs))
}

/// Change it in place: its name, when it runs (in words) and what its runs
/// do (a reminder's message stays). Saved changes apply from its next run.
fn edit_form(
    store: Store,
    id: String,
    name: String,
    edit: aidaemon_client_protocol::AutomationEdit,
) -> View {
    let new_name = signal(name.clone());
    let new_when = signal(edit.when.clone());
    let instructions = edit.instructions.clone();
    let new_instructions = signal(instructions.clone().unwrap_or_default());
    let notify_was = edit.notify;
    let new_notify = signal(notify_was);
    let changed = |now: String, was: &str| {
        let now = now.trim().to_owned();
        (!now.is_empty() && now != was.trim()).then_some(now)
    };
    let save = {
        let (name, when, instructions) = (name.clone(), edit.when.clone(), instructions.clone());
        move || {
            store.edit_automation(
                &id,
                changed(new_name.get_untracked(), &name),
                changed(new_when.get_untracked(), &when),
                instructions
                    .as_deref()
                    .and_then(|was| changed(new_instructions.get_untracked(), was)),
                new_notify
                    .get_untracked()
                    .filter(|now| Some(*now) != notify_was),
            )
        }
    };
    let unchanged = {
        let (name, when, instructions) = (name, edit.when, instructions.clone());
        move || {
            changed(new_name.get(), &name).is_none()
                && changed(new_when.get(), &when).is_none()
                && instructions
                    .as_deref()
                    .is_none_or(|was| changed(new_instructions.get(), was).is_none())
                && new_notify.get() == notify_was
        }
    };
    let mut form = view()
        .gap_2()
        .pt_2()
        .child(ui::labeled(
            "Name",
            ui::field("Name", new_name).label("Name"),
        ))
        .child(ui::labeled(
            "When it runs",
            ui::field("Every weekday at 9am", new_when).label("When it runs"),
        ));
    if instructions.is_some() {
        form = form.child(ui::labeled(
            "What each run does",
            ui::area("What each run does", new_instructions).label("What it does"),
        ));
    }
    // Quiet unless it matters: what its runs tell you (a reminder always
    // does). A quiet run is still in its history.
    if notify_was.is_some() {
        use aidaemon_client_protocol::AutomationNotify as Notify;
        let key = |mode: Notify| match mode {
            Notify::EveryRun => "every_run",
            Notify::Changes => "changes",
            Notify::Problems => "problems",
            Notify::Unknown => "",
        };
        form = form.child(ui::labeled(
            "Tell me",
            ui::choice(
                "Tell me",
                "Every run",
                vec![
                    ("every_run".to_owned(), "Every run".to_owned()),
                    (
                        "changes".to_owned(),
                        "Only when something changes".to_owned(),
                    ),
                    ("problems".to_owned(), "Only about problems".to_owned()),
                ],
                move || new_notify.get().map(|mode| key(mode).to_owned()),
                move |chosen| {
                    new_notify.set(match chosen.as_deref() {
                        Some("changes") => Some(Notify::Changes),
                        Some("problems") => Some(Notify::Problems),
                        _ => Some(Notify::EveryRun),
                    })
                },
            ),
        ));
    }
    form.child(
        text("Changes apply from its next run.")
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT),
    )
    .child(
        view()
            .flex_row()
            .gap_1()
            .child(ui::btn("Save", Variant::Primary, Size::Sm, save).disabled(unchanged))
            .child(ui::btn("Discard", Variant::Ghost, Size::Sm, move || {
                store.set_automation_editing(None)
            })),
    )
}
