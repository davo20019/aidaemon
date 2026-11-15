//! The details pane beside a conversation: for a coding conversation its
//! Changes (`features::code`) and Steps, for any other its Steps alone:
//! the work running for it and what its current turn did, step by step.
//! What waits on the owner across conversations is the sidebar's Needs
//! you, not here.

use nsh_core::prelude::*;

use aidaemon_client_protocol::TaskSummary;

use crate::store::{ActivityKind, CHANGES_TAB, DETAIL_TABS, Store, now_ms};
use crate::ui::{self, Glyph, Size, Tone, Variant, btn, colors, radius, section_label, type_scale};

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
                    view().flex_1().child(
                        when(coding, move || {
                            ui::tab_bar(store.inspector_tab(), &DETAIL_TABS)
                        })
                        .otherwise(|| {
                            text("Steps")
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
            let changes = coding.get() && store.inspector_tab().get() == CHANGES_TAB;
            Some(untracked(move || {
                if changes {
                    super::code::code(store)
                } else {
                    steps(store)
                }
            }))
        }))
}

/// The Steps tab: the selected conversation's running work, then its
/// current turn.
fn steps(store: Store) -> View {
    let running = computed(move || {
        let selected = store.selected();
        store
            .running()
            .into_iter()
            .filter(|t| Some(&t.session_id) == selected.as_ref())
            .collect::<Vec<TaskSummary>>()
    });
    view().flex_1().child(
        scroll().flex_1().child(
            view()
                .px_3()
                .pb_4()
                .gap_2()
                .child(when(
                    move || !running.with(Vec::is_empty),
                    move || {
                        view().gap_1().child(section_label("Running")).child(each(
                            running,
                            |t| t.id.clone(),
                            move |task, _| {
                                let id = task.with_untracked(|t| t.id.clone());
                                let started = task.with_untracked(|t| t.started_at_ms);
                                view()
                                    .flex_row()
                                    .items_center()
                                    .gap_2()
                                    .px_2()
                                    .py_1()
                                    .child(ui::icon(Glyph::Work).text_color(colors::ACCENT))
                                    .child(
                                        text(move || task.with(|t| t.description.clone()))
                                            .flex_1()
                                            .min_w(px(0.0))
                                            .number_of_lines(1)
                                            .text_size(px(type_scale::SM)),
                                    )
                                    // How long it has run: a clock tick a second.
                                    .child(
                                        text(move || {
                                            store.tick();
                                            ui::time::span(now_ms().saturating_sub(started) / 1000)
                                        })
                                        .font_family(ui::theme::MONO)
                                        .text_size(px(type_scale::XS))
                                        .text_color(colors::FAINT),
                                    )
                                    .child(btn("Cancel", Variant::Ghost, Size::Sm, move || {
                                        store.cancel(&id)
                                    }))
                            },
                        ))
                    },
                ))
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
                text("Tool calls and steps of the current turn show here as they happen.")
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
                                    let output = kind == ActivityKind::Output;
                                    let all = signal(false);
                                    let lines = move || item.with(|a| a.detail.lines().count());
                                    // Long output shows its end (where errors are), with
                                    // Show all and Copy.
                                    let shown = move || {
                                        let text = detail();
                                        if !output || all.get() {
                                            return text;
                                        }
                                        let lines: Vec<&str> = text.lines().collect();
                                        let skip = lines.len().saturating_sub(8);
                                        let tail = lines[skip..].join("\n");
                                        if skip > 0 {
                                            format!("… {skip} lines above\n{tail}")
                                        } else {
                                            tail
                                        }
                                    };
                                    view()
                                        .gap_1()
                                        .when(output, |v| {
                                            v.rounded(px(radius::SMALL))
                                                .bg(colors::INSET)
                                                .px_2()
                                                .py_1()
                                        })
                                        .child(
                                            text(shown)
                                                .number_of_lines(if output { 0 } else { 3 })
                                                .selectable(true)
                                                .when(output, |t| t.font_family(ui::theme::MONO))
                                                .text_size(px(type_scale::XS))
                                                .text_color(colors::MUTED),
                                        )
                                        .child(when(
                                            move || output && lines() > 8,
                                            move || {
                                                view()
                                                    .flex_row()
                                                    .gap_1()
                                                    .child(dynamic(move || {
                                                        let label = if all.get() {
                                                            "Show less"
                                                        } else {
                                                            "Show all"
                                                        };
                                                        Some(untracked(move || {
                                                            btn(
                                                                label,
                                                                Variant::Ghost,
                                                                Size::Sm,
                                                                move || all.update(|a| *a = !*a),
                                                            )
                                                        }))
                                                    }))
                                                    .child(btn(
                                                        "Copy",
                                                        Variant::Ghost,
                                                        Size::Sm,
                                                        move || set_clipboard(detail()),
                                                    ))
                                            },
                                        ))
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
