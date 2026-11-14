//! The sidebar: a new conversation, the conversations with what each is
//! doing, and the connection.

use nsh_core::prelude::*;

use crate::store::{Connection, Page, SearchRead, Store, now_ms};
use crate::ui::{self, Glyph, Tone, colors, icon, row, section_label, size, status, type_scale};
use aidaemon_client_core::machine::DropKind;

pub fn sidebar(store: Store) -> View {
    let ids = computed(move || store.visible_ids());
    // Grouped by what each needs: an answer, then work running, then the rest.
    let needs = computed(move || {
        ids.with(|ids| {
            ids.iter()
                .filter(|id| store.needs_you(id))
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    let working = computed(move || {
        ids.with(|ids| {
            ids.iter()
                .filter(|id| !store.needs_you(id) && store.is_running(id))
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    let rest = computed(move || {
        ids.with(|ids| {
            ids.iter()
                .filter(|id| !store.needs_you(id) && !store.is_running(id))
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    let query = signal(store.filter());
    effect(move || {
        let q = query.get();
        untracked(|| {
            store.set_filter(q.clone());
            // Every conversation's messages, not only the ones loaded here.
            store.search_messages(&q);
        });
    });
    view()
        .w(px(size::SIDEBAR))
        .flex_1()
        .style_when(move || store.is_compact(), |s| s.w_full())
        .bg(colors::SIDEBAR)
        .border_r_1()
        .border_color(colors::LINE)
        .child(
            view()
                .h(px(ui::fit::bar()))
                .flex_row()
                .items_center()
                .px_3()
                .gap_2()
                .child(
                    text("AIdaemon")
                        .flex_1()
                        .role(AccessibilityRole::Header)
                        .text_size(px(ui::fit::body()))
                        .font_weight(FontWeight::SEMIBOLD),
                )
                .child(
                    when(
                        move || store.is_compact(),
                        move || {
                            ui::icon_btn(ui::Glyph::Close, "Close", ui::Size::Md, move || {
                                store.close_panes()
                            })
                        },
                    )
                    .otherwise(|| ui::kbd(key_label("mod+k"))),
                ),
        )
        .child(
            view().px_2().child(
                row(
                    move || store.selected().is_none(),
                    move || store.select(None),
                )
                .accessibility_label("New conversation")
                .child(icon(Glyph::Plus).text_color(colors::MUTED))
                .child(
                    text("New conversation")
                        .flex_1()
                        .text_size(px(type_scale::BASE)),
                )
                .child(ui::kbd(key_label("mod+n"))),
            ),
        )
        .child(
            view().px_2().pt_2().child(
                ui::field("Search conversations", query)
                    .label("Search conversations")
                    .w_full(),
            ),
        )
        .child(
            scroll().flex_1().child(
                view()
                    .px_2()
                    .pb_2()
                    .gap(px(1.0))
                    .child(group(store, "Needs you", needs))
                    .child(group(store, "Working", working))
                    .child(when(
                        move || !rest.with(Vec::is_empty),
                        move || {
                            section_label(move || {
                                if needs.with(Vec::is_empty) && working.with(Vec::is_empty) {
                                    "Conversations"
                                } else {
                                    "Others"
                                }
                            })
                        },
                    ))
                    .child(each(
                        rest,
                        |id: &String| id.clone(),
                        move |id, _| session_row(store, id.get_untracked()),
                    ))
                    .child(search_hits(store))
                    .child(when(
                        move || ids.with(Vec::is_empty) && !store.filter().trim().is_empty(),
                        || {
                            text("No conversation matches.")
                                .px_2()
                                .py_2()
                                .text_size(px(type_scale::SM))
                                .text_color(colors::FAINT)
                        },
                    )),
            ),
        )
        .child(
            view().px_2().pb_1().child(
                row(
                    move || store.page() == Page::System,
                    move || store.open_system(),
                )
                .accessibility_label("System: how the daemon is doing")
                .child(icon(Glyph::System).text_color(colors::MUTED))
                .child(text("System").flex_1().text_size(px(ui::fit::body())))
                .child(
                    text(move || {
                        store
                            .today_tokens()
                            .filter(|n| *n > 0)
                            .map(|n| format!("{} today", ui::time::compact_count(n)))
                            .unwrap_or_default()
                    })
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
                ),
            ),
        )
        .child(
            view().px_2().pb_1().child(
                row(
                    move || store.page() == Page::Memory,
                    move || store.open_memory(),
                )
                .accessibility_label("Memory: what the daemon remembers")
                .child(icon(Glyph::Memory).text_color(colors::MUTED))
                .child(text("Memory").flex_1().text_size(px(ui::fit::body()))),
            ),
        )
        .child(ui::separator())
        .child(connection(store))
}

fn session_row(store: Store, id: String) -> nsh_base::ButtonElement {
    let (a, b, c, d, e) = (id.clone(), id.clone(), id.clone(), id.clone(), id.clone());
    let (f, g, h) = (id.clone(), id.clone(), id.clone());
    let label_id = id.clone();
    row(
        move || store.selected().as_deref() == Some(a.as_str()),
        move || store.select(Some(b.clone())),
    )
    // Read as one item: its name, where it lives and whether it is working.
    .accessibility_label(move || {
        let id = &label_id;
        let mut parts = vec![store.title(id), store.place(id)];
        if store.is_pinned(id) {
            parts.push("pinned".into());
        }
        if store.is_running(id) {
            parts.push("working".into());
        }
        if store.is_unread(id) {
            parts.push("new reply".into());
        }
        Some(SharedString::from(parts.join(", ")))
    })
    // Working (accent, pulsing), a new reply (filled) or quiet (hollow).
    .child(
        text(move || {
            if store.is_running(&c) || store.is_unread(&c) {
                "●"
            } else {
                "○"
            }
        })
        .w(px(size::ICON))
        .text_center()
        .text_size(px(8.0))
        .text_color(move || {
            if store.is_running(&e) {
                colors::ACCENT
            } else if store.is_unread(&e) {
                colors::TEXT
            } else {
                colors::FAINT
            }
        }),
    )
    .child(
        view()
            .flex_1()
            .min_w(px(0.0))
            .child(
                view()
                    .flex_row()
                    .items_center()
                    .gap_1()
                    .child(
                        text(move || store.title(&d))
                            .flex_1()
                            .number_of_lines(1)
                            .text_size(px(ui::fit::body()))
                            .text_color(move || {
                                if store.is_unread(&h) {
                                    colors::TEXT
                                } else {
                                    colors::MUTED
                                }
                            }),
                    )
                    .child(
                        text(move || if store.is_pinned(&f) { "★" } else { "" })
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                    )
                    .child(
                        text(move || {
                            store
                                .last_activity(&g)
                                .map(|at| ui::time::ago(at, now_ms()))
                                .unwrap_or_default()
                        })
                        .text_size(px(type_scale::XS))
                        .text_color(colors::FAINT),
                    ),
            )
            .child(
                text(move || store.place(&id))
                    .number_of_lines(1)
                    .text_size(px(if ui::touch() { 12.0 } else { type_scale::XS }))
                    .text_color(colors::FAINT),
            ),
    )
}

/// A titled group of conversations, shown only when it has any.
fn group(store: Store, title: &'static str, ids: Memo<Vec<String>>) -> View {
    view()
        .gap(px(1.0))
        .child(when(
            move || !ids.with(Vec::is_empty),
            move || section_label(title),
        ))
        .child(each(
            ids,
            |id: &String| id.clone(),
            move |id, _| session_row(store, id.get_untracked()),
        ))
}

/// Messages matching the search, from every conversation the daemon has.
fn search_hits(store: Store) -> View {
    view().gap(px(1.0)).child(dynamic(move || {
        let read = store.search();
        Some(untracked(move || match read {
            SearchRead::Idle => view(),
            SearchRead::Loading(_) => view().child(section_label("In messages: searching…")),
            SearchRead::Failed(error) => view().child(
                text(error)
                    .px_2()
                    .text_size(px(type_scale::SM))
                    .text_color(colors::MUTED),
            ),
            SearchRead::Ready { hits, .. } => {
                let mut list = view().gap(px(1.0)).child(section_label(if hits.is_empty() {
                    "In messages: nothing"
                } else {
                    "In messages"
                }));
                for hit in hits {
                    let open = hit.session_id.clone();
                    let who = if hit.role == "user" {
                        "You"
                    } else {
                        "AIdaemon"
                    };
                    let title = store.title(&hit.session_id);
                    list = list.child(
                        row(|| false, move || store.select(Some(open.clone())))
                            .accessibility_label(format!("{title}: {}", hit.snippet))
                            .child(
                                view()
                                    .flex_1()
                                    .min_w(px(0.0))
                                    .py_1()
                                    .child(
                                        text(hit.snippet.clone())
                                            .number_of_lines(2)
                                            .text_size(px(type_scale::SM))
                                            .text_color(colors::TEXT),
                                    )
                                    .child(
                                        text(format!(
                                            "{who} · {title} · {}",
                                            ui::time::ago(hit.at_ms, now_ms())
                                        ))
                                        .number_of_lines(1)
                                        .text_size(px(type_scale::XS))
                                        .text_color(colors::FAINT),
                                    ),
                            ),
                    );
                }
                list
            }
        }))
    }))
}

fn connection(store: Store) -> View {
    view().px_4().py_3().child(dynamic(move || {
        let state = store.connection();
        Some(untracked(move || match state {
            Connection::Online => status("Connected to the daemon", Tone::Success),
            Connection::Connecting => status("Connecting…", Tone::Accent),
            Connection::NeedsPairing => status("Not paired yet", Tone::Warning),
            Connection::Offline {
                kind,
                reason,
                retry_at_ms,
            } => {
                let headline = match kind {
                    DropKind::DaemonNotRunning => "The daemon is not running",
                    DropKind::Unauthorized => "The daemon refused this app",
                    DropKind::VersionUnsupported => "The daemon needs a newer app",
                    DropKind::Dropped => "Connection lost",
                };
                let retry = retry_at_ms.saturating_sub(now_ms()) / 1000;
                view().gap_1().child(status(headline, Tone::Danger)).child(
                    text(if kind == DropKind::Unauthorized {
                        reason
                    } else {
                        format!("Retrying in {}s", retry.max(1))
                    })
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
                )
            }
        }))
    }))
}
