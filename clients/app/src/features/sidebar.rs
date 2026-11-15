//! The sidebar: a new conversation, the conversations that need the owner
//! or are working (from every project), then each project with its
//! conversations, the conversations in no project (filterable by where
//! they came from: the app, each chat platform, groups), and the
//! connection.

use nsh_core::prelude::*;

use crate::store::{Connection, Page, SearchRead, Source, Store, now_ms};
use crate::ui::{self, Glyph, Tone, colors, icon, row, section_label, status, type_scale};
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
                .filter(|id| !store.needs_you(id) && store.is_working(id))
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    let rest = computed(move || {
        ids.with(|ids| {
            ids.iter()
                .filter(|id| !store.needs_you(id) && !store.is_working(id))
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
        .w_full()
        .flex_1()
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
                // On a phone, New conversation is a button here: the list
                // keeps the room a full row would take.
                .child(when(ui::touch(), move || {
                    ui::icon_btn(Glyph::Plus, "New conversation", ui::Size::Md, move || {
                        store.select(None)
                    })
                }))
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
        .child(when(!ui::touch(), move || {
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
            )
        }))
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
                    // Searching lists every match flat; otherwise by project.
                    .child(
                        when(
                            move || store.filter().trim().is_empty(),
                            move || projects(store, rest),
                        )
                        .otherwise(move || group(store, "Conversations", rest)),
                    )
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
        .child(pages(store))
        // On a phone the line shows only when something is wrong: connected
        // is the normal state, not news worth a row.
        .child(
            view()
                .style_when(
                    move || ui::touch() && store.connection() == Connection::Online,
                    |s| s.hidden(),
                )
                .child(ui::separator())
                .child(connection(store)),
        )
}

/// A conversation's row, with its menu on a right click or a long press:
/// open it, pin or unpin it, stop its work, move it to a project.
fn session_row(store: Store, id: String) -> View {
    let (open, pin, stop) = (id.clone(), id.clone(), id.clone());
    let pinned = untracked(|| store.is_pinned(&id));
    let current = untracked(|| store.session_project(&id));
    let mut actions = vec![
        ui::MenuAction::new("Open", move || store.select(Some(open.clone()))),
        ui::MenuAction::new(if pinned { "Unpin" } else { "Pin" }, move || {
            store.toggle_pin(&pin)
        }),
        {
            let working = stop.clone();
            ui::MenuAction::new("Stop its work", move || store.stop(&stop))
                .shown_when(move || store.is_working(&working))
        },
    ];
    for project in untracked(|| store.projects()) {
        if current.as_deref() == Some(project.id.as_str()) {
            continue;
        }
        let (session, target) = (id.clone(), project.id.clone());
        actions.push(ui::MenuAction::new(
            format!("Move to {}", project.name),
            move || store.move_to_project(&session, Some(target.clone())),
        ));
    }
    if current.is_some() {
        let session = id.clone();
        actions.push(ui::MenuAction::new("Take out of its project", move || {
            store.move_to_project(&session, None)
        }));
    }
    let preview = id.clone();
    ui::with_menu(
        ui::with_preview(session_button(store, id), move || {
            session_preview(store, preview)
        }),
        actions,
    )
}

/// More about a conversation than its row says: its title, where it is,
/// when it last moved, and its last message.
fn session_preview(store: Store, id: String) -> View {
    let (title, place, when, last) = (id.clone(), id.clone(), id.clone(), id);
    view()
        .gap_1()
        .child(
            text(move || store.title(&title))
                .number_of_lines(2)
                .text_size(px(type_scale::SM))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(colors::TEXT),
        )
        .child(
            text(move || {
                let mut parts = vec![store.place(&place)];
                if let Some(at) = store.last_activity(&when) {
                    parts.push(ui::time::ago(at, crate::store::now_ms()));
                }
                parts.retain(|p| !p.is_empty());
                parts.join(" · ")
            })
            .number_of_lines(1)
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT),
        )
        .child(
            text(move || match store.last_message(&last) {
                Some((true, text)) => format!("You: {}", text.trim()),
                Some((false, text)) => text.trim().to_owned(),
                None => "Open it to read it.".to_owned(),
            })
            .number_of_lines(4)
            .text_size(px(type_scale::XS))
            .text_color(colors::MUTED),
        )
}

/// The projects, each folded open or shut with its conversations, then
/// the conversations in no project. `rest`: the conversations not already
/// listed above (needing the owner, or working).
fn projects(store: Store, rest: Memo<Vec<String>>) -> View {
    let all = computed(move || store.projects());
    let unplaced = computed(move || {
        let known: Vec<String> = all.with(|all| all.iter().map(|p| p.id.clone()).collect());
        rest.with(|ids| {
            ids.iter()
                .filter(|id| {
                    store
                        .session_project(id)
                        .is_none_or(|project| !known.contains(&project))
                        && store.in_source(id)
                })
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    view()
        .gap(px(1.0))
        .child(
            view()
                .flex_row()
                .items_center()
                .child(section_label("Projects").flex_1())
                .child(ui::icon_btn(
                    Glyph::Plus,
                    "New project",
                    ui::Size::Sm,
                    move || store.show_project_form(!store.project_form_open()),
                )),
        )
        .child(when(
            move || store.project_form_open(),
            move || project_form(store),
        ))
        .child(when(
            move || all.with(Vec::is_empty) && !store.project_form_open(),
            || {
                text("Group conversations by folder: a site, a repository, designs.")
                    .px_2()
                    .pb_1()
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT)
            },
        ))
        .child(each(
            all,
            |p| p.id.clone(),
            move |project, _| {
                let id = project.with_untracked(|p| p.id.clone());
                project_group(store, id, rest)
            },
        ))
        .child(when(
            move || !unplaced.with(Vec::is_empty) || store.source_filter() != Source::All,
            move || section_label("Conversations"),
        ))
        .child(sources(store))
        .child(when(
            move || unplaced.with(Vec::is_empty) && store.source_filter() != Source::All,
            || {
                text("None from here.")
                    .px_2()
                    .py_1()
                    .text_size(px(type_scale::SM))
                    .text_color(colors::FAINT)
            },
        ))
        .child(each(
            unplaced,
            |id: &String| id.clone(),
            move |id, _| session_row(store, id.get_untracked()),
        ))
}

/// Where the conversations came from, as filters: shown only when they
/// came from more than one place.
fn sources(store: Store) -> View {
    view().child(dynamic(move || {
        let sources = store.sources();
        Some(untracked(move || {
            let mut chips = view().flex_row().flex_wrap().gap_1().px_2().pb_1();
            for source in sources {
                chips = chips.child(ui::chip(
                    source.label(),
                    move || store.source_filter() == source,
                    move || store.set_source_filter(source),
                ));
            }
            chips
        }))
    }))
}

/// One project: its row (fold, name, folder, New conversation) and, open,
/// its conversations. Its menu edits it.
fn project_group(store: Store, id: String, rest: Memo<Vec<String>>) -> View {
    let mine = {
        let id = id.clone();
        computed(move || {
            rest.with(|ids| {
                ids.iter()
                    .filter(|s| store.session_project(s).as_deref() == Some(id.as_str()))
                    .cloned()
                    .collect::<Vec<_>>()
            })
        })
    };
    let (toggle, open, new, name, folder, label, empty_id) = (
        id.clone(),
        id.clone(),
        id.clone(),
        id.clone(),
        id.clone(),
        id.clone(),
        id.clone(),
    );
    let is_open = computed(move || store.project_open(&open));
    let (start, edit) = (id.clone(), id.clone());
    let actions = vec![
        ui::MenuAction::new("New conversation here", move || {
            store.new_in_project(Some(start.clone()))
        }),
        ui::MenuAction::new("Edit name or folder", move || store.edit_project(&edit)),
    ];
    view()
        .gap(px(1.0))
        .child(ui::with_menu(
            view()
                .flex_row()
                .items_center()
                .child(
                    row(
                        move || {
                            store.selected().is_none()
                                && store.current_project().as_deref() == Some(new.as_str())
                        },
                        move || store.toggle_project(&toggle),
                    )
                    .flex_1()
                    .min_w(px(0.0))
                    .accessibility_label(move || {
                        let project = store.project(&label);
                        Some(SharedString::from(format!(
                            "Project {}, {} conversations",
                            project.map(|p| p.name).unwrap_or_default(),
                            mine.with(Vec::len)
                        )))
                    })
                    .child(
                        icon(Glyph::Chevron)
                            .text_color(colors::FAINT)
                            .rotate(move || deg(if is_open.get() { 90.0 } else { 0.0 }))
                            .transition_transform()
                            .duration(ui::motion::FAST),
                    )
                    .child(
                        view()
                            .flex_1()
                            .min_w(px(0.0))
                            .child(
                                text(move || {
                                    store.project(&name).map(|p| p.name).unwrap_or_default()
                                })
                                .number_of_lines(1)
                                .text_size(px(ui::fit::body()))
                                .font_weight(FontWeight::MEDIUM),
                            )
                            .child(
                                text(move || {
                                    store
                                        .project(&folder)
                                        .and_then(|p| p.path)
                                        .map(|path| short_path(&path))
                                        .unwrap_or_else(|| "No folder".to_owned())
                                })
                                .number_of_lines(1)
                                .text_size(px(type_scale::XS))
                                .text_color(colors::FAINT),
                            ),
                    )
                    .child(
                        text(move || match mine.with(Vec::len) {
                            0 => String::new(),
                            n => n.to_string(),
                        })
                        .text_size(px(type_scale::XS))
                        .text_color(colors::FAINT),
                    ),
                )
                .child(ui::icon_btn(
                    Glyph::Plus,
                    "New conversation in this project",
                    ui::Size::Sm,
                    {
                        let id = id.clone();
                        move || store.new_in_project(Some(id.clone()))
                    },
                )),
            actions,
        ))
        .child(when(
            move || is_open.get(),
            move || {
                let empty_id = empty_id.clone();
                view()
                    .pl_3()
                    .gap(px(1.0))
                    .child(when(
                        move || mine.with(Vec::is_empty),
                        move || {
                            let id = empty_id.clone();
                            row(|| false, move || store.new_in_project(Some(id.clone()))).child(
                                text("No conversations yet. Start one")
                                    .text_size(px(type_scale::SM))
                                    .text_color(colors::FAINT),
                            )
                        },
                    ))
                    .child(each(
                        mine,
                        |id: &String| id.clone(),
                        move |id, _| session_row(store, id.get_untracked()),
                    ))
            },
        ))
}

/// The project form, under the Projects heading: a new project, or the
/// one being edited.
fn project_form(store: Store) -> View {
    let editing = untracked(|| store.project_editing().and_then(|id| store.project(&id)));
    let name = signal(editing.as_ref().map(|p| p.name.clone()).unwrap_or_default());
    let path = signal(
        editing
            .as_ref()
            .and_then(|p| p.path.clone())
            .unwrap_or_default(),
    );
    let saving_label = if editing.is_some() {
        "Save"
    } else {
        "Create project"
    };
    view()
        .px_2()
        .py_2()
        .gap_2()
        // The form opens to be typed in (a phone keeps its keyboard down).
        .child(
            ui::field("Name", name)
                .label("Project name")
                .auto_focus(!ui::touch())
                .w_full(),
        )
        .child(
            ui::field("Folder, e.g. ~/projects/acme-site", path)
                .label("Project folder on the daemon's computer")
                .w_full(),
        )
        .child(
            text("Any folder: a site, a repository, designs. Leave it empty for none.")
                .text_size(px(type_scale::XS))
                .text_color(colors::FAINT),
        )
        .child(when(
            move || store.project_error().is_some(),
            move || {
                ui::banner(
                    move || store.project_error().unwrap_or_default(),
                    Tone::Danger,
                )
            },
        ))
        .child(
            view()
                .flex_row()
                .gap_2()
                .child(ui::btn(
                    saving_label,
                    ui::Variant::Primary,
                    ui::Size::Sm,
                    move || store.save_project(&name.get_untracked(), &path.get_untracked()),
                ))
                .child(ui::btn(
                    "Cancel",
                    ui::Variant::Ghost,
                    ui::Size::Sm,
                    move || store.show_project_form(false),
                )),
        )
}

/// A folder for a row: the home folder as `~`, and only its last parts
/// when long.
pub fn short_path(path: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let shown = match path.strip_prefix(&home) {
        Some(rest) if !home.is_empty() => format!("~{rest}"),
        _ => path.to_owned(),
    };
    let parts: Vec<&str> = shown.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() > 3 {
        format!("…/{}", parts[parts.len() - 2..].join("/"))
    } else {
        shown
    }
}

fn session_button(store: Store, id: String) -> nsh_base::ButtonElement {
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
    .child(ui::state_dot(
        move || store.is_running(&c) || store.is_unread(&c),
        move || {
            if store.is_running(&e) {
                colors::ACCENT
            } else if store.is_unread(&e) {
                colors::TEXT
            } else {
                colors::FAINT
            }
        },
    ))
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

/// Automations, System, Memory and Settings: rows on a desktop; on a phone
/// one row of four, so the conversations keep the screen.
fn pages(store: Store) -> View {
    if ui::touch() {
        let tab = |glyph: Glyph,
                   label: &'static str,
                   name: &'static str,
                   page: Page,
                   open: fn(&Store)| {
            ui::row(move || store.page() == page, move || open(&store))
                .flex_1()
                .flex_col()
                .justify_center()
                .gap(px(2.0))
                .accessibility_label(name)
                .child(icon(glyph).text_color(colors::MUTED))
                .child(
                    text(label)
                        .number_of_lines(1)
                        .text_size(px(type_scale::XS))
                        .text_color(colors::MUTED),
                )
        };
        return view()
            .flex_row()
            .px_2()
            .py_1()
            .gap_1()
            .border_t_1()
            .border_color(colors::LINE)
            .child(tab(
                Glyph::Clock,
                "Automate",
                "Automations: what runs on its own",
                Page::Automations,
                Store::open_automations,
            ))
            .child(tab(
                Glyph::System,
                "System",
                "System: how the daemon is doing",
                Page::System,
                Store::open_system,
            ))
            .child(tab(
                Glyph::Memory,
                "Memory",
                "Memory: what the daemon remembers",
                Page::Memory,
                Store::open_memory,
            ))
            .child(tab(
                Glyph::Settings,
                "Settings",
                "Settings: change how the daemon runs",
                Page::Settings,
                Store::open_settings,
            ));
    }
    view()
        .child(
            view().px_2().pb_1().child(
                row(
                    move || store.page() == Page::Automations,
                    move || store.open_automations(),
                )
                .accessibility_label("Automations: what runs on its own")
                .child(icon(Glyph::Clock).text_color(colors::MUTED))
                .child(text("Automations").flex_1().text_size(px(ui::fit::body()))),
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
        .child(
            view().px_2().pb_1().child(
                row(
                    move || store.page() == Page::Settings,
                    move || store.open_settings(),
                )
                .accessibility_label("Settings: change how the daemon runs")
                .child(icon(Glyph::Settings).text_color(colors::MUTED))
                .child(text("Settings").flex_1().text_size(px(ui::fit::body()))),
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
    view().px_4().py_2().child(dynamic(move || {
        let state = store.connection();
        Some(untracked(move || match state {
            Connection::Online => status("Connected to the daemon", Tone::Success),
            Connection::Connecting => status("Connecting…", Tone::Accent),
            Connection::NeedsPairing => status("Not paired yet", Tone::Warning),
            Connection::NoDaemon => status("No daemon yet", Tone::Warning),
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
                // One row: the footer sits under the conversation list and
                // must not eat its room on a phone.
                let detail = if kind == DropKind::Unauthorized {
                    reason
                } else {
                    let retry = (retry_at_ms.saturating_sub(now_ms()) / 1000).max(1);
                    format!("retrying in {retry}s")
                };
                view()
                    .flex_row()
                    .items_center()
                    .gap_2()
                    .child(status(headline, Tone::Danger).flex_1().min_w(px(0.0)))
                    .child(
                        text(detail)
                            .number_of_lines(1)
                            .text_size(px(type_scale::XS))
                            .text_color(colors::FAINT),
                    )
            }
        }))
    }))
}
