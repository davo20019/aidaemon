//! The Automations page: everything that runs on its own (scheduled work
//! and reminders, watching automations, health checks), next to run first,
//! each with Pause or Resume, Run now (scheduled work) and Cancel (asked
//! twice). New ones are made by asking the agent, so the page starts a
//! conversation for that. Read on open, every 30 seconds while open, on
//! Refresh, and after each change (its answer brings the list).

use aidaemon_client_protocol::{AutomationChange, AutomationKind, AutomationSummary};
use nsh_core::prelude::*;

use crate::store::{AutomationsRead, Store, now_ms};
use crate::ui::{self, Glyph, Size, Tone, Variant, badge, colors, size, time, type_scale};

pub fn page(store: Store) -> View {
    set_interval(std::time::Duration::from_secs(30), move || {
        store.refresh_automations()
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
        .child(ui::btn("New", Variant::Ghost, Size::Sm, move || {
            new_automation(store)
        }))
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
            let mut column = column;
            for automation in list {
                column = column.child(card(store, automation));
            }
            column.child(
                text("To make one or change when it runs, ask in a conversation.")
                    .px_2()
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
            )
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
        "awaiting_input" => Some(("Waiting for you", Tone::Warning)),
        "pending_confirmation" => Some(("Not confirmed yet", Tone::Warning)),
        _ => Some(("Off", Tone::Neutral)),
    }
}

fn card(store: Store, automation: AutomationSummary) -> View {
    let id = automation.id.clone();
    let title = automation
        .name
        .clone()
        .unwrap_or_else(|| automation.what.clone());
    let mut schedule = format!("{} · {}", kind_word(automation.kind), automation.how_often);
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
    let mut heading = view().flex_row().items_center().gap_2().child(
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
    let mut c = ui::surface().p_3().gap_1().child(heading).child(
        text(schedule)
            .text_size(px(type_scale::XS))
            .text_color(colors::MUTED),
    );
    if automation.name.is_some() {
        c = c.child(
            text(automation.what.clone())
                .number_of_lines(3)
                .text_size(px(type_scale::SM))
                .text_color(colors::MUTED),
        );
    }
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
    let mut actions = view().flex_row().flex_wrap().gap_1().pt_1();
    if !waiting {
        actions = actions.child(
            if paused {
                ui::btn(
                    "Resume",
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
    if automation.kind == AutomationKind::Scheduled {
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
    let cancel = act(AutomationChange::Cancel);
    actions = actions.child(
        when(confirming, move || {
            let cancel = cancel.clone();
            view()
                .flex_row()
                .items_center()
                .gap_1()
                .child(
                    text("Cancel it for good?")
                        .text_size(px(type_scale::SM))
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
            ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                confirming.set(true)
            })
        }),
    );
    c.child(actions)
}
