//! The Agents tab: a conversation's participants, one row per coding agent
//! session that worked in it (`aidaemon_client_core::participants`). Two
//! Claude Code runs started apart are two rows; a follow-up joins the row
//! of the session it continues. A row says who it is, what it is doing or
//! how it ended, whether it can take a follow-up and when it last acted,
//! and opens its thread, where a follow-up goes to it. The owner can name
//! it ("Designer"): every client shows the name the daemon keeps.

use aidaemon_client_core::participants::Participant;
use aidaemon_client_protocol::RunPhase;
use nsh_base::button;
use nsh_core::prelude::*;

use crate::store::Store;
use crate::ui::{self, Glyph, colors, section_label, type_scale};

pub fn participants(store: Store) -> View {
    let list = computed(move || store.participants());
    view().flex_1().child(
        scroll().flex_1().child(
            view()
                .px_3()
                .pb_4()
                .gap_2()
                .child(section_label(move || match list.with(Vec::len) {
                    1 => "1 agent".to_string(),
                    n => format!("{n} agents"),
                }))
                .child(each(list, |p| p.thread.clone(), move |p, _| row(store, p))),
        ),
    )
}

/// What it is doing, or how it ended and whether it takes a follow-up.
pub fn state_words(p: &Participant) -> &'static str {
    if p.waiting {
        return "waiting on you";
    }
    match p.latest.phase {
        RunPhase::Running => "working",
        _ if p.can_continue() => "can continue",
        RunPhase::Succeeded => "finished",
        RunPhase::Failed => "failed",
        RunPhase::Stopped if p.latest.stopped_by_owner => "stopped by you",
        RunPhase::Stopped => "stopped",
        RunPhase::Unknown => "",
    }
}

fn row(store: Store, p: ReadSignal<Participant>) -> View {
    let editing = signal(false);
    view()
        .w_full()
        .child(when(
            move || !editing.get(),
            move || shown(store, p, editing),
        ))
        .child(when(
            move || editing.get(),
            move || rename_field(store, p, editing),
        ))
}

/// Its name as a field: Return or Save keeps it, Cancel leaves it; an
/// empty name goes back to what it was started for.
fn rename_field(store: Store, p: ReadSignal<Participant>, editing: Signal<bool>) -> View {
    let name = signal(p.with_untracked(|p| {
        if p.named {
            p.name.clone()
        } else {
            String::new()
        }
    }));
    let save = move || {
        let (session, thread) =
            p.with_untracked(|p| (p.latest.session_id.clone(), p.thread.clone()));
        store.rename_agent(&session, &thread, name.get_untracked());
        editing.set(false);
    };
    view()
        .px_2()
        .py_1()
        .gap_1()
        .child(
            ui::field("Designer, Tester…", name)
                .label("Agent name")
                .auto_focus(true)
                .on_submit(move |_: &str| save())
                .w_full(),
        )
        .child(
            view()
                .flex_row()
                .gap_1()
                .child(ui::btn("Save", ui::Variant::Primary, ui::Size::Sm, save))
                .child(ui::btn(
                    "Cancel",
                    ui::Variant::Ghost,
                    ui::Size::Sm,
                    move || editing.set(false),
                )),
        )
}

fn shown(store: Store, p: ReadSignal<Participant>, editing: Signal<bool>) -> View {
    let mark = computed(move || {
        p.with(|p| match p.latest.phase {
            _ if p.waiting => (Glyph::Approval, colors::WARNING),
            RunPhase::Running => (Glyph::Work, colors::ACCENT),
            RunPhase::Failed => (Glyph::Warning, colors::DANGER),
            RunPhase::Stopped => (Glyph::Warning, colors::WARNING),
            _ if p.can_continue() => (Glyph::Dot, colors::SUCCESS),
            _ => (Glyph::Check, colors::MUTED),
        })
    });
    let title = computed(move || p.with(Participant::title));
    let detail = computed(move || {
        store.tick();
        p.with(|p| {
            let mut words = state_words(p).to_string();
            if p.working() && !p.waiting && !p.latest.step.is_empty() {
                words = format!("{words} · {}", p.latest.step);
            }
            if p.runs > 1 {
                words.push_str(&format!(" · {} runs", p.runs));
            }
            let ago = ui::time::ago(p.last_activity_ms, crate::store::now_ms());
            if !p.working() && !ago.is_empty() {
                words.push_str(&format!(" · {ago}"));
            }
            words
        })
    });
    view()
        .w_full()
        .flex_row()
        .items_center()
        .gap_1()
        .child(
            button()
                .flex_1()
                .min_w(px(0.0))
                .accessibility_label(move || {
                    Some(SharedString::from(format!(
                        "{}, {}, open its thread",
                        title.get(),
                        detail.get()
                    )))
                })
                .on_press(move || store.open_run(&p.with_untracked(|p| p.latest.id.clone())))
                .child(
                    view()
                        .w_full()
                        .flex_row()
                        .items_start()
                        .gap_2()
                        .px_2()
                        .py_1p5()
                        .rounded(px(ui::radius::CONTROL))
                        .hover(|s| s.bg(colors::HOVER))
                        .child(dynamic(move || {
                            let (glyph, color) = mark.get();
                            Some(untracked(move || {
                                ui::icon(glyph).text_color(color).mt(px(2.0))
                            }))
                        }))
                        .child(
                            view()
                                .flex_1()
                                .min_w(px(0.0))
                                .gap(px(2.0))
                                .child(
                                    text(move || title.get())
                                        .number_of_lines(1)
                                        .text_size(px(type_scale::SM))
                                        .font_weight(FontWeight::MEDIUM),
                                )
                                .child(
                                    text(move || detail.get())
                                        .number_of_lines(1)
                                        .text_size(px(type_scale::XS))
                                        .text_color(colors::FAINT),
                                ),
                        ),
                ),
        )
        .child(ui::icon_btn(
            Glyph::Pen,
            "Rename this agent",
            ui::Size::Sm,
            move || editing.set(true),
        ))
}
