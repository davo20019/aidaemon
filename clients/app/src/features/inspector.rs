//! The details pane beside a conversation: for a coding conversation its
//! Changes (`features::code`) and Steps, for any other its Steps alone:
//! the work running for it and what its current turn did, step by step.
//! What waits on the owner across conversations is the sidebar's Needs
//! you, not here.

use nsh_core::prelude::*;

use crate::store::{ActivityKind, CHANGES_TAB, DETAIL_TABS, Store};
use crate::ui::{self, Glyph, Size, Tone, Variant, btn, colors, section_label, type_scale};

pub fn inspector(store: Store) -> View {
    let coding = computed(move || store.coding_on_screen());
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
                .px_4()
                .child(
                    // Only a coding conversation has this pane. A phone
                    // shows its changes alone (the steps are under each
                    // reply, and a pane over the chat hides them as they
                    // run); a desktop has Steps beside them, to watch a
                    // turn's commands and output next to the chat.
                    view().flex_1().child(
                        when(
                            move || coding.get() && !store.is_compact(),
                            move || ui::tab_bar(store.inspector_tab(), &DETAIL_TABS),
                        )
                        .otherwise(move || {
                            text(move || {
                                if store.is_compact() {
                                    "Changes"
                                } else {
                                    "Steps"
                                }
                            })
                            .role(AccessibilityRole::Header)
                            .text_size(px(ui::fit::body()))
                            .font_weight(FontWeight::SEMIBOLD)
                        }),
                    ),
                )
                .child(when(
                    move || store.is_compact(),
                    move || {
                        ui::icon_btn(ui::Glyph::Close, "Close", ui::Size::Md, move || {
                            store.close_panes()
                        })
                    },
                )),
        )
        .child(dynamic(move || {
            let changes =
                coding.get() && (store.is_compact() || store.inspector_tab().get() == CHANGES_TAB);
            Some(untracked(move || {
                if changes {
                    super::code::code(store)
                } else {
                    steps(store)
                }
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
