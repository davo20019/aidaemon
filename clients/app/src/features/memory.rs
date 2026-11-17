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
            let count = text(match shown.len() {
                0 => format!("Nothing matches \"{query}\""),
                1 => "1 memory".to_owned(),
                n => format!("{n} memories"),
            })
            .px_1()
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT);
            // Grouped by category, each group where its newest memory
            // falls (the list arrives newest first), rows newest first.
            let mut groups: Vec<(String, Vec<MemoryItem>)> = Vec::new();
            for item in shown {
                match groups.iter_mut().find(|g| g.0 == item.category) {
                    Some(group) => group.1.push(item),
                    None => groups.push((item.category.clone(), vec![item])),
                }
            }
            let mut page = view().gap_2().child(count);
            for (category, items) in groups {
                let heading = match readable_key(&category) {
                    name if name.is_empty() => "Other".to_owned(),
                    name => name,
                };
                let mut card = ui::surface().overflow_hidden();
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        card = card.child(ui::separator());
                    }
                    card = card.child(memory_row(store, item.clone()));
                }
                page = page.child(
                    view()
                        .gap_1()
                        .child(
                            section_label(format!("{heading} · {}", items.len()))
                                .role(AccessibilityRole::Header),
                        )
                        .child(card),
                );
            }
            page
        }
    }
}

/// Where a memory came from, in the owner's words.
fn source_words(source: &str) -> Option<String> {
    Some(
        match source {
            "" => return None,
            "user_stated" | "owner_quoted" => "you said it",
            "consolidation" => "from past conversations",
            "progressive" => "picked up while talking",
            "inferred" | "derived" => "inferred",
            "agent" => "saved by the assistant",
            "task_learning" => "learned from a task",
            other => return Some(other.replace('_', " ")),
        }
        .to_owned(),
    )
}

fn memory_row(store: Store, item: MemoryItem) -> View {
    let id = item.id.clone();
    let forgotten = format!("Your agent stops remembering: {}", item.value);
    let label = readable_key(&item.key);
    let meta = {
        let mut parts = vec![time::ago(item.updated_ms, now_ms())];
        if let Some(source) = source_words(&item.source) {
            parts.push(source);
        }
        parts.join(" · ")
    };
    // The key and Forget share the top line, so what is remembered gets
    // the card's whole width under them.
    let top = view()
        .flex_row()
        .items_center()
        .gap_2()
        .child(
            text(if label.is_empty() {
                "Memory".to_owned()
            } else {
                label
            })
            .flex_1()
            .min_w(px(0.0))
            .number_of_lines(1)
            .text_size(px(ui::fit::small()))
            .font_weight(FontWeight::SEMIBOLD)
            .text_color(colors::MUTED),
        )
        .child(
            ui::btn("Forget", Variant::Ghost, Size::Sm, move || {
                let id = id.clone();
                ui::confirm("Forget this?", forgotten.clone(), "Forget", move || {
                    store.forget_memory(&id)
                })
            })
            .accessibility_label(format!("Forget: {}", item.value)),
        );
    view().pl_3().pr_1().pb_2().child(top).child(
        view()
            .pr_2()
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
}

/// A fact's key as words, so a bare value such as `ddev` says what it is
/// ("Local dev tool"). The model sees the key; the owner should too.
fn readable_key(key: &str) -> String {
    let words = key.replace(['_', '-'], " ");
    let words = words.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = words.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}
