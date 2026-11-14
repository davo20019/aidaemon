//! The inspector: everything waiting on the owner (approvals across all
//! conversations), the work running now, and what the current turn did,
//! step by step.

use nsh_core::prelude::*;

use super::approvals;
use crate::store::{ActivityKind, Store};
use crate::ui::{
    self, Glyph, Size, Tone, Variant, badge, btn, colors, radius, section_label, size, type_scale,
};

pub fn inspector(store: Store) -> View {
    let approvals = computed(move || store.approvals());
    let running = computed(move || store.running());
    view()
        .w(px(size::INSPECTOR))
        .flex_1()
        .style_when(move || store.is_compact(), |s| s.w_full())
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
                    text("Activity")
                        .flex_1()
                        .role(AccessibilityRole::Header)
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::SEMIBOLD),
                )
                .child(when(
                    move || approvals.with(|a| !a.is_empty()),
                    move || {
                        badge(
                            move || format!("{} waiting", approvals.with(Vec::len)),
                            Tone::Warning,
                        )
                    },
                ))
                .child(when(
                    move || store.is_compact(),
                    move || {
                        ui::icon_btn(ui::Glyph::Close, "Close", ui::Size::Md, move || {
                            store.close_panes()
                        })
                    },
                )),
        )
        .child(
            scroll().flex_1().child(
                view()
                    .px_3()
                    .pb_4()
                    .gap_2()
                    // Approvals of other conversations: the current one shows
                    // its own inline.
                    .child(when(
                        move || {
                            let current = store.selected();
                            approvals
                                .with(|a| a.iter().any(|a| Some(&a.session_id) != current.as_ref()))
                        },
                        move || {
                            view()
                                .gap_2()
                                .child(section_label("Waiting for you"))
                                .child(each(
                                    move || {
                                        let current = store.selected();
                                        approvals.with(|a| {
                                            a.iter()
                                                .filter(|a| Some(&a.session_id) != current.as_ref())
                                                .cloned()
                                                .collect::<Vec<_>>()
                                        })
                                    },
                                    |a| a.id.clone(),
                                    move |a, _| {
                                        let approval = a.get_untracked();
                                        let session = approval.session_id.clone();
                                        view()
                                            .gap_1()
                                            .child(
                                                ui::row(
                                                    || false,
                                                    move || store.select(Some(session.clone())),
                                                )
                                                .child(
                                                    text(move || {
                                                        format!(
                                                            "In {}",
                                                            store.title(
                                                                &a.with(|a| a.session_id.clone())
                                                            )
                                                        )
                                                    })
                                                    .flex_1()
                                                    .number_of_lines(1)
                                                    .text_size(px(type_scale::XS))
                                                    .text_color(colors::MUTED),
                                                ),
                                            )
                                            .child(approvals::card(store, approval))
                                    },
                                ))
                        },
                    ))
                    .child(section_label("Running"))
                    .child(
                        when(
                            move || running.with(Vec::is_empty),
                            || {
                                text("Nothing running.")
                                    .px_2()
                                    .text_size(px(type_scale::SM))
                                    .text_color(colors::FAINT)
                            },
                        )
                        .otherwise(move || {
                            view().gap_1().child(each(
                                running,
                                |t| t.id.clone(),
                                move |task, _| {
                                    let id = task.with_untracked(|t| t.id.clone());
                                    let session = task.with_untracked(|t| t.session_id.clone());
                                    view()
                                        .flex_row()
                                        .items_center()
                                        .gap_2()
                                        .px_2()
                                        .py_1()
                                        .child(ui::icon(Glyph::Work).text_color(colors::ACCENT))
                                        .child(
                                            ui::row(
                                                || false,
                                                move || store.select(Some(session.clone())),
                                            )
                                            .flex_1()
                                            .min_w(px(0.0))
                                            .child(
                                                text(move || task.with(|t| t.description.clone()))
                                                    .flex_1()
                                                    .number_of_lines(1)
                                                    .text_size(px(type_scale::SM)),
                                            ),
                                        )
                                        .child(btn("Cancel", Variant::Ghost, Size::Sm, move || {
                                            store.cancel(&id)
                                        }))
                                },
                            ))
                        }),
                    )
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
                                    view()
                                        .when(output, |v| {
                                            v.rounded(px(radius::SMALL))
                                                .bg(colors::INSET)
                                                .px_2()
                                                .py_1()
                                        })
                                        .child(
                                            text(detail)
                                                .number_of_lines(if output { 8 } else { 3 })
                                                .selectable(true)
                                                .when(output, |t| t.font_family(ui::theme::MONO))
                                                .text_size(px(type_scale::XS))
                                                .text_color(colors::MUTED),
                                        )
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
