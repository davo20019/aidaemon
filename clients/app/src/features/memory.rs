//! The Memory page: what the daemon remembers about the owner, newest
//! first, each with Forget. Read on open and on Refresh.

use aidaemon_client_protocol::MemoryItem;
use nsh_core::prelude::*;

use crate::store::{MemoryRead, Store, now_ms};
use crate::ui::{self, Glyph, Size, Tone, Variant, colors, section_label, size, time, type_scale};

pub fn page(store: Store) -> View {
    let filter = signal(String::new());
    view().flex_1().min_w(px(0.0)).child(header(store)).child(
        scroll().flex_1().child(
            view().items_center().child(
                view()
                    .w_full()
                    .max_w(px(size::COLUMN))
                    .px_4()
                    .py_4()
                    .gap_3()
                    .child(
                        text(
                            "What your assistant remembers about you and uses in its answers. \
                                 Forget anything that is wrong or that it should not keep.",
                        )
                        .text_size(px(ui::fit::small()))
                        .text_color(colors::MUTED),
                    )
                    .child(
                        ui::field("Filter memories", filter)
                            .label("Filter memories")
                            .w_full(),
                    )
                    .child(dynamic(move || {
                        let read = store.memories();
                        let query = filter.get().trim().to_lowercase();
                        Some(untracked(move || body(store, read, &query)))
                    })),
            ),
        ),
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
            text("Memory")
                .flex_1()
                .px_2()
                .role(AccessibilityRole::Header)
                .text_size(px(ui::fit::body()))
                .font_weight(FontWeight::SEMIBOLD),
        )
        .child(ui::icon_btn(Glyph::Retry, "Refresh", Size::Md, move || {
            store.refresh_memories()
        }))
}

fn body(store: Store, read: MemoryRead, query: &str) -> View {
    match read {
        MemoryRead::NotYet | MemoryRead::Loading => view().child(ui::working("Reading memories…")),
        MemoryRead::Failed(reason) => view().child(ui::banner(reason, Tone::Danger).child(
            ui::btn("Try again", Variant::Secondary, Size::Sm, move || {
                store.refresh_memories()
            }),
        )),
        MemoryRead::Ready(items) if items.is_empty() => view().child(ui::empty(
            "Nothing remembered yet",
            "As you talk, the assistant keeps facts and preferences here.",
        )),
        MemoryRead::Ready(items) => {
            let shown: Vec<MemoryItem> = items
                .into_iter()
                .filter(|m| {
                    query.is_empty()
                        || format!("{} {} {}", m.category, m.key, m.value)
                            .to_lowercase()
                            .contains(query)
                })
                .collect();
            let mut list = view().gap_1().child(section_label(format!(
                "{} {}",
                shown.len(),
                if shown.len() == 1 {
                    "memory"
                } else {
                    "memories"
                }
            )));
            for item in shown {
                list = list.child(memory_row(store, item));
            }
            list
        }
    }
}

fn memory_row(store: Store, item: MemoryItem) -> View {
    let id = item.id.clone();
    let meta = {
        let mut parts = vec![item.category.clone()];
        if !item.source.is_empty() {
            parts.push(item.source.clone());
        }
        parts.push(format!("updated {}", time::ago(item.updated_ms, now_ms())));
        parts.join(" · ")
    };
    ui::surface()
        .flex_row()
        .items_center()
        .gap_3()
        .px_3()
        .py_2()
        .child(
            view()
                .flex_1()
                .min_w(px(0.0))
                .gap(px(2.0))
                .child(
                    text(item.value.clone())
                        .selectable(true)
                        .text_size(px(ui::fit::body()))
                        .text_color(colors::TEXT),
                )
                .child(
                    text(meta)
                        .number_of_lines(1)
                        .text_size(px(type_scale::XS))
                        .text_color(colors::FAINT),
                ),
        )
        .child(
            ui::btn("Forget", Variant::Ghost, Size::Sm, move || {
                store.forget_memory(&id)
            })
            .accessibility_label(format!("Forget: {}", item.value)),
        )
}
