//! The Memory page: what the daemon remembers about the owner, newest
//! first, with review filtering, inline editing, reversible archiving,
//! and permanent Forget. Read on open and on Refresh.

use std::collections::HashMap;

use aidaemon_client_core::MemoriesState;
use aidaemon_client_protocol::MemoryItem;
use nsh_core::prelude::*;

use crate::store::{MemoryRead, Store, now_ms};
use crate::ui::{self, Size, Tone, Variant, colors, section_label, size, time};

#[derive(Clone, Copy, PartialEq, Eq)]
enum MemoryTab {
    All = 0,
    Review = 1,
    Archived = 2,
}

impl MemoryTab {
    fn from_index(index: usize) -> Self {
        match index {
            1 => Self::Review,
            2 => Self::Archived,
            _ => Self::All,
        }
    }
}

pub fn page(store: Store) -> View {
    let filter = signal(String::new());
    let tab = signal(MemoryTab::All);
    let editing_id: Signal<Option<String>> = signal(None);
    let edit_draft = signal(String::new());
    view()
        .flex_1()
        .min_w(px(0.0))
        .child(header(store))
        .child(ui::refreshable(
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
                                 Edit, archive, or forget anything that is wrong or should not stay.",
                            )
                            .text_size(px(ui::fit::small()))
                            .text_color(colors::MUTED),
                        )
                        .child(ui::segmented(
                            "Memory view",
                            &["All", "Needs review", "Archived"],
                            move || tab.get() as usize,
                            move |idx| {
                                editing_id.set(None);
                                tab.set(MemoryTab::from_index(idx));
                            },
                        ))
                        .child(
                            ui::field("Filter memories", filter)
                                .label("Filter memories")
                                .w_full(),
                        )
                        .child(dynamic(move || {
                            let read = store.memories();
                            let can_manage = store.can_manage_memories();
                            let query = filter.get().trim().to_lowercase();
                            let current_tab = tab.get();
                            let editing = editing_id.get();
                            Some(untracked(move || {
                                body(
                                    store,
                                    read,
                                    can_manage,
                                    &query,
                                    current_tab,
                                    editing.as_deref(),
                                    editing_id,
                                    edit_draft,
                                )
                            }))
                        })),
                ),
            ),
            move || store.pull_refreshing(),
            move || store.pull_refresh(),
        ))
}

fn header(store: Store) -> View {
    crate::features::config_nav::header(store)
}

fn is_review_candidate(item: &MemoryItem) -> bool {
    if item.overlaps_with.is_some() {
        return true;
    }
    let base = item
        .source
        .strip_prefix("canonical:")
        .unwrap_or(&item.source);
    matches!(base, "inferred" | "derived")
}

fn matches_query(m: &MemoryItem, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }
    let excerpt = m.excerpt.as_deref().unwrap_or("");
    format!("{} {} {} {}", m.category, m.key, m.value, excerpt)
        .to_lowercase()
        .contains(query)
}

#[allow(clippy::too_many_arguments)]
fn body(
    store: Store,
    read: MemoryRead,
    can_manage: bool,
    query: &str,
    tab: MemoryTab,
    editing: Option<&str>,
    editing_id: Signal<Option<String>>,
    edit_draft: Signal<String>,
) -> View {
    match read {
        MemoryRead::NotYet | MemoryRead::Loading => view().child(ui::working("Reading memories…")),
        MemoryRead::Failed(reason) => view().child(ui::banner(reason, Tone::Danger).child(
            ui::btn("Try again", Variant::Secondary, Size::Sm, move || {
                store.refresh_memories()
            }),
        )),
        MemoryRead::Ready(state) => render_ready(
            store, state, can_manage, query, tab, editing, editing_id, edit_draft,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn render_ready(
    store: Store,
    state: MemoriesState,
    can_manage: bool,
    query: &str,
    tab: MemoryTab,
    editing: Option<&str>,
    editing_id: Signal<Option<String>>,
    edit_draft: Signal<String>,
) -> View {
    if state.items.is_empty() && state.archived.is_empty() {
        return view().child(ui::empty(
            "Nothing remembered yet",
            "As you talk, the assistant keeps facts and preferences here.",
        ));
    }

    let review_total = state
        .items
        .iter()
        .filter(|m| is_review_candidate(m))
        .count();
    let by_id: HashMap<String, (String, String)> = state
        .items
        .iter()
        .map(|m| (m.id.clone(), (readable_key(&m.key), m.value.clone())))
        .collect();

    let source_items: &[MemoryItem] = match tab {
        MemoryTab::All | MemoryTab::Review => &state.items,
        MemoryTab::Archived => &state.archived,
    };

    let shown: Vec<MemoryItem> = source_items
        .iter()
        .filter(|m| tab != MemoryTab::Review || is_review_candidate(m))
        .filter(|m| matches_query(m, query))
        .cloned()
        .collect();

    if shown.is_empty() && query.is_empty() {
        let (title, detail) = match tab {
            MemoryTab::Review => (
                "Nothing needs review",
                "No inferred memories or possible duplicates right now.",
            ),
            MemoryTab::Archived => (
                "No archived memories",
                "Memories you archive leave active recall and stay here until restored.",
            ),
            MemoryTab::All => (
                "No active memories",
                "Archived memories are still available under Archived.",
            ),
        };
        return view().child(ui::empty(title, detail));
    }

    let summary_text = match tab {
        MemoryTab::All => {
            if !query.is_empty() {
                match shown.len() {
                    0 => format!("Nothing matches \"{query}\""),
                    1 => "1 matching memory".to_owned(),
                    n => format!("{n} matching memories"),
                }
            } else {
                let active_total = state.total.unwrap_or(state.items.len() as u64) as usize;
                let base = if active_total > state.items.len() {
                    format!("Showing {} of {} memories", state.items.len(), active_total)
                } else if active_total == 1 {
                    "1 memory".to_owned()
                } else {
                    format!("{active_total} memories")
                };
                if review_total > 0 {
                    format!("{base} · {review_total} to review")
                } else {
                    base
                }
            }
        }
        MemoryTab::Review => match shown.len() {
            0 => format!("Nothing matches \"{query}\""),
            1 => "1 memory to review".to_owned(),
            n => format!("{n} memories to review"),
        },
        MemoryTab::Archived => match shown.len() {
            0 => format!("Nothing matches \"{query}\""),
            1 => "1 archived memory".to_owned(),
            n => format!("{n} archived memories"),
        },
    };

    let count = text(summary_text)
        .px_1()
        .text_size(px(ui::fit::meta()))
        .text_color(colors::FAINT);

    // Grouped by case-normalized category, each group where its newest memory
    // falls (the list arrives newest first), rows newest first.
    let mut groups: Vec<(String, Vec<MemoryItem>)> = Vec::new();
    for item in shown {
        let cat_key = item.category.trim().to_ascii_lowercase();
        match groups.iter_mut().find(|g| g.0 == cat_key) {
            Some(group) => group.1.push(item),
            None => groups.push((cat_key, vec![item])),
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
            let overlap_hint = item
                .overlaps_with
                .as_ref()
                .and_then(|other_id| by_id.get(other_id))
                .map(|(other_key, other_val)| {
                    format!("Similar to {}: {}", other_key, preview_text(other_val, 80))
                });
            let is_editing = can_manage && editing == Some(item.id.as_str());
            card = card.child(memory_row(
                store,
                item.clone(),
                can_manage,
                overlap_hint,
                is_editing,
                editing_id,
                edit_draft,
            ));
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

/// Where a memory came from, in the owner's words.
fn source_words(source: &str) -> Option<String> {
    let base = source.strip_prefix("canonical:").unwrap_or(source);
    Some(
        match base {
            "" => return None,
            "user_stated" | "owner" | "owner-stated" => "you said it",
            "owner_quoted" => "quoted from chat",
            "consolidation" => "from past conversations",
            "progressive" => "picked up while talking",
            "inferred" | "derived" => "inferred",
            "agent" => "saved by the assistant",
            "task_learning" => "learned from a task",
            "legacy" => "saved earlier",
            other => return Some(other.replace('_', " ")),
        }
        .to_owned(),
    )
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let single = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if single.chars().count() <= max_chars {
        single
    } else {
        let cut: String = single.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{}…", cut.trim_end())
    }
}

#[allow(clippy::too_many_arguments)]
fn memory_row(
    store: Store,
    item: MemoryItem,
    can_manage: bool,
    overlap_hint: Option<String>,
    is_editing: bool,
    editing_id: Signal<Option<String>>,
    edit_draft: Signal<String>,
) -> View {
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

    let mut title_row = view()
        .flex_1()
        .min_w(px(0.0))
        .flex_row()
        .flex_wrap()
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
        );
    if overlap_hint.is_some() {
        title_row = title_row.child(ui::badge("Duplicate?", Tone::Warning));
    }

    let mut actions = view().flex_row().flex_wrap().items_center().gap_1();
    if item.archived {
        if can_manage {
            let restore_id = id.clone();
            actions = actions.child(
                ui::btn("Restore", Variant::Secondary, Size::Sm, move || {
                    store.restore_memory(&restore_id)
                })
                .accessibility_label(format!("Restore: {}", item.value)),
            );
        }
    } else {
        if can_manage && item.editable && !is_editing {
            let start_id = id.clone();
            let current_value = item.value.clone();
            actions = actions.child(
                ui::btn("Edit", Variant::Ghost, Size::Sm, move || {
                    edit_draft.set(current_value.clone());
                    editing_id.set(Some(start_id.clone()));
                })
                .accessibility_label(format!("Edit: {}", item.value)),
            );
        }
        if can_manage && item.kind.as_deref() != Some("canonical") && !is_editing {
            let archive_id = id.clone();
            actions = actions.child(
                ui::btn("Archive", Variant::Ghost, Size::Sm, move || {
                    store.archive_memory(&archive_id)
                })
                .accessibility_label(format!("Archive: {}", item.value)),
            );
        }
    }
    if !is_editing {
        let forget_id = id.clone();
        actions = actions.child(
            ui::btn("Forget", Variant::Ghost, Size::Sm, move || {
                let forget_id = forget_id.clone();
                ui::confirm("Forget this?", forgotten.clone(), "Forget", move || {
                    store.forget_memory(&forget_id)
                })
            })
            .accessibility_label(format!("Forget: {}", item.value)),
        );
    }

    let top = view()
        .flex_row()
        .flex_wrap()
        .items_center()
        .gap_2()
        .child(title_row)
        .child(actions);

    let mut body_col = view().pr_2().gap(px(3.0));
    if is_editing {
        let submit_id = id.clone();
        let save_id = id.clone();
        body_col = body_col
            .child(
                ui::field("Memory value", edit_draft)
                    .label("Memory value")
                    .w_full()
                    .on_submit(move |_: &str| {
                        let next = edit_draft.get().trim().to_owned();
                        if !next.is_empty() {
                            store.edit_memory(&submit_id, &next);
                            editing_id.set(None);
                        }
                    }),
            )
            .child(
                view()
                    .flex_row()
                    .items_center()
                    .gap_2()
                    .child(dynamic(move || {
                        let empty = edit_draft.get().trim().is_empty();
                        let save_id = save_id.clone();
                        Some(
                            ui::btn("Save", Variant::Primary, Size::Sm, move || {
                                let next = edit_draft.get().trim().to_owned();
                                if !next.is_empty() {
                                    store.edit_memory(&save_id, &next);
                                    editing_id.set(None);
                                }
                            })
                            .disabled(empty),
                        )
                    }))
                    .child(ui::btn("Cancel", Variant::Ghost, Size::Sm, move || {
                        editing_id.set(None);
                    })),
            );
    } else {
        body_col = body_col.child(
            text(item.value.clone())
                .selectable(true)
                .text_size(px(ui::fit::body()))
                .text_color(colors::TEXT),
        );
    }

    if let Some(hint) = overlap_hint {
        body_col = body_col.child(
            text(hint)
                .number_of_lines(2)
                .text_size(px(ui::fit::meta()))
                .text_color(colors::MUTED),
        );
    }

    if let Some(excerpt) = item.excerpt.as_deref()
        && !excerpt.eq_ignore_ascii_case(item.value.trim())
    {
        body_col = body_col.child(
            text(format!("“{}”", preview_text(excerpt, 140)))
                .number_of_lines(2)
                .text_size(px(ui::fit::meta()))
                .text_color(colors::FAINT),
        );
    }

    body_col = body_col.child(
        text(meta)
            .number_of_lines(1)
            .text_size(px(ui::fit::meta()))
            .text_color(colors::FAINT),
    );

    view().pl_3().pr_1().pb_2().child(top).child(body_col)
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
