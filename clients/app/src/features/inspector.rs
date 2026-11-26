//! The details pane beside a conversation: for a coding conversation its
//! Changes (`features::code`) and its project's Files (`features::files`);
//! for any, its Steps (the work running for it and what its current turn
//! did, step by step), its Terminal (`features::terminal`, the shell it
//! shares with the agent), its Browser (`features::browser_tab`) and its
//! Agents (`features::participants`: each coding agent session in it).
//! What waits on the owner across conversations is the sidebar's Needs
//! you, not here.

use nsh_core::prelude::*;

use crate::store::{
    AGENTS_TAB, ActivityKind, BROWSER_TAB, CHANGES_TAB, FILES_TAB, STEPS_TAB, Store, TERMINAL_TAB,
    detail_tabs,
};
use crate::ui::{self, Glyph, Size, Tone, Variant, btn, colors, section_label, type_scale};

pub fn inspector(store: Store) -> View {
    view().w_full().flex_1().child(dynamic(move || {
        // A run's view, opened from its row, takes the pane until closed.
        let run = store.open_run_id();
        Some(untracked(move || match run {
            Some(id) => super::run_view::run_view(store, id),
            None => tabs(store),
        }))
    }))
}

fn tabs(store: Store) -> View {
    let coding = computed(move || store.coding_on_screen());
    let agents = computed(move || store.has_participants());
    // A medium window's drawer (narrower than the page's side pane): a
    // phone's room, so a phone's tabs and its Close.
    let drawer = computed(move || store.details_drawer() && window().width.0 < 1180.0);
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
                // The tabs stand on the bar's line, level with the
                // conversation header's beside it.
                .items_end()
                .px_4()
                .border_b_1()
                .border_color(colors::LINE)
                .child(
                    // Code has its changes and files; a desktop has Steps
                    // beside the chat (a phone has them under each reply);
                    // every conversation has its terminal and browser.
                    view().flex_1().child(dynamic(move || {
                        let tabs = detail_tabs(
                            coding.get(),
                            store.is_compact() || drawer.get(),
                            agents.get(),
                        );
                        Some(untracked(move || {
                            ui::bar_tabs_of(store.inspector_tab(), &tabs)
                        }))
                    })),
                )
                .child(when(
                    // A phone's, or a medium window's drawer.
                    move || store.is_compact() || drawer.get(),
                    move || {
                        view().self_center().child(ui::icon_btn(
                            ui::Glyph::Close,
                            "Close",
                            ui::Size::Md,
                            move || store.close_panes(),
                        ))
                    },
                )),
        )
        .child(dynamic(move || {
            // A tab this conversation does not show falls back to its first.
            let tabs = detail_tabs(
                coding.get(),
                store.is_compact() || drawer.get(),
                agents.get(),
            );
            let picked = store.inspector_tab().get();
            let tab = if tabs.iter().any(|(i, _)| *i == picked) {
                picked
            } else {
                tabs.first().map_or(STEPS_TAB, |(i, _)| *i)
            };
            Some(untracked(move || match tab {
                CHANGES_TAB => super::code::code(store),
                FILES_TAB => super::files::files(store),
                TERMINAL_TAB => super::terminal::terminal_tab(store),
                BROWSER_TAB => super::browser_tab::browser_tab(store),
                AGENTS_TAB => super::participants::participants(store),
                _ => steps(store),
            }))
        }))
}

/// The Steps tab: the current turn's steps as they happen, its commands'
/// output in a terminal. (Work running in the background is over the
/// composer.)
fn steps(store: Store) -> View {
    view().flex_1().child(
        scroll().flex_1().child(
            view()
                .px_3()
                .pb_4()
                .gap_2()
                .child(section_label("This turn"))
                .child(timeline(store)),
        ),
    )
}

fn glyph(kind: ActivityKind) -> (Glyph, Tone) {
    match kind {
        ActivityKind::Thinking => (Glyph::Dot, Tone::Neutral),
        ActivityKind::Tool => (Glyph::Tool, Tone::Accent),
        ActivityKind::ToolDone => (Glyph::Check, Tone::Success),
        ActivityKind::Output => (Glyph::Chevron, Tone::Neutral),
        ActivityKind::Plan => (Glyph::Dot, Tone::Info),
        ActivityKind::Note => (Glyph::Dot, Tone::Neutral),
    }
}

fn timeline(store: Store) -> View {
    let items = computed(move || store.inspected_turn().map(|t| t.items).unwrap_or_default());
    let done = computed(move || store.inspected_turn().is_some_and(|t| t.done));
    view()
        .gap(px(2.0))
        .child(when(
            move || items.with(Vec::is_empty),
            || {
                text("No turn is running. Each reply lists the tools it used under it.")
                    .px_2()
                    .text_size(px(type_scale::SM))
                    .text_color(colors::FAINT)
            },
        ))
        .child(each(
            items,
            |a| a.id,
            move |item, _| {
                let kind = item.with_untracked(|a| a.kind);
                let (g, tone) = glyph(kind);
                let detail = move || item.with(|a| a.detail.clone());
                view()
                    .flex_row()
                    .gap_2()
                    .px_2()
                    .py_1()
                    .child(ui::icon(g).text_color(tone.color()))
                    .child(
                        view()
                            .flex_1()
                            .min_w(px(0.0))
                            .gap(px(2.0))
                            .child(
                                text(move || item.with(|a| a.title.clone()))
                                    .number_of_lines(1)
                                    .font_family(ui::theme::MONO)
                                    .text_size(px(type_scale::SM)),
                            )
                            .child(when(
                                move || item.with(|a| !a.detail.is_empty()),
                                move || {
                                    if kind == ActivityKind::Output {
                                        ui::output_panel(detail, "Step output")
                                    } else {
                                        step_detail(detail)
                                    }
                                },
                            )),
                    )
            },
        ))
        .child(when(done, || {
            text("Turn finished.")
                .px_2()
                .pt_1()
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT)
        }))
}

/// A step's detail: three lines, and More (a button, so the keyboard
/// reaches it too) for the rest when there is more.
fn step_detail(detail: impl Fn() -> String + Clone + 'static) -> View {
    let all = signal(false);
    let long = {
        let detail = detail.clone();
        move || {
            let text = detail();
            text.lines().count() > 3 || text.chars().count() > 240
        }
    };
    view()
        .gap_1()
        .child(
            text(detail)
                .number_of_lines(move || if all.get() { 0_u32 } else { 3_u32 })
                .selectable(true)
                .text_size(px(type_scale::XS))
                .text_color(colors::MUTED),
        )
        .child(when(long, move || {
            view().flex_row().child(dynamic(move || {
                let label = if all.get() { "Less" } else { "More" };
                Some(untracked(move || {
                    btn(label, Variant::Ghost, Size::Sm, move || {
                        all.update(|a| *a = !*a)
                    })
                }))
            }))
        }))
}
