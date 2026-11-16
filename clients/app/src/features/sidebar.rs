//! The sidebar. Home: search and one New conversation, the conversations
//! that need the owner or are working, the pinned ones, the projects used
//! last (one line each; a press goes inside one), then every conversation
//! by day, tagged with its project or the chat it came from. A project's
//! page: its folder, its conversations, New conversation there, Edit.
//! Then the pages and, when something is wrong, the connection.

use nsh_core::prelude::*;

use crate::store::{Connection, Page, SearchRead, SidebarView, Source, Store, now_ms};
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
    let searching = move || !store.filter().trim().is_empty();
    view()
        .w_full()
        .flex_1()
        .bg(colors::SIDEBAR)
        .border_r_1()
        .border_color(colors::LINE)
        .child(when(ui::touch(), move || {
            // A phone: search, one New conversation and Close on one line.
            view()
                .min_h(px(ui::fit::bar()))
                .flex_row()
                .items_center()
                .px_2()
                .gap_1()
                .child(
                    ui::field("Search", query)
                        .label("Search conversations")
                        .flex_1(),
                )
                .child(ui::icon_btn(
                    Glyph::Compose,
                    "New conversation",
                    ui::Size::Md,
                    move || new_here(store),
                ))
                .child(when(
                    move || store.is_compact(),
                    move || {
                        ui::icon_btn(Glyph::Close, "Close", ui::Size::Md, move || {
                            store.close_panes()
                        })
                    },
                ))
        }))
        .child(when(!ui::touch(), move || {
            view()
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
                                    ui::icon_btn(Glyph::Close, "Close", ui::Size::Md, move || {
                                        store.close_panes()
                                    })
                                },
                            )
                            .otherwise(|| ui::kbd(key_label("mod+k"))),
                        ),
                )
                .child(
                    view().px_2().child(
                        row(move || store.selected().is_none(), move || new_here(store))
                            .accessibility_label("New conversation")
                            .child(icon(Glyph::Compose).text_color(colors::MUTED))
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
        }))
        .child(
            scroll().flex_1().child(
                view()
                    .px_2()
                    .pb_2()
                    .gap(px(1.0))
                    .child(dynamic(move || {
                        // Searching lists every match flat, and so do the
                        // archived; otherwise the view the owner is in.
                        let flat = searching() || store.showing_archived();
                        let shown = store.sidebar_view();
                        Some(untracked(move || {
                            if flat {
                                return view()
                                    .gap(px(1.0))
                                    .child(group(store, "Needs you", needs))
                                    .child(group(store, "Working", working))
                                    .child(group(store, "Conversations", rest));
                            }
                            match shown {
                                SidebarView::Home => home(store, needs, working, rest),
                                SidebarView::Projects => all_projects(store, ids),
                                SidebarView::Project(id) => project_page(store, id, ids),
                            }
                        }))
                    }))
                    .child(search_hits(store))
                    .child(archived_toggle(store))
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
fn session_row(store: Store, id: String, dated: bool) -> View {
    let editing = id.clone();
    let shown = id.clone();
    // Rename shows the name as a field in the row's place.
    view().child(
        when(
            move || store.renaming().as_deref() == Some(editing.as_str()),
            move || rename_field(store, id.clone()),
        )
        .otherwise(move || menu_row(store, shown.clone(), dated)),
    )
}

/// The row with its menu.
fn menu_row(store: Store, id: String, dated: bool) -> View {
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
    // One line however many projects there are: the picker lists them,
    // with No project first to take it out of one.
    if untracked(|| !store.projects().is_empty()) {
        let session = id.clone();
        actions.push(ui::MenuAction::new(
            if current.is_some() {
                "Move to another project…"
            } else {
                "Move to project…"
            },
            move || store.start_moving(&session),
        ));
    }
    {
        let session = id.clone();
        actions.push(ui::MenuAction::new("Rename", move || {
            store.set_renaming(Some(session.clone()))
        }));
    }
    {
        let session = id.clone();
        let archived = untracked(|| store.is_archived(&id));
        actions.push(ui::MenuAction::new(
            if archived { "Bring back" } else { "Archive" },
            move || store.archive_session(&session, !archived),
        ));
    }
    let preview = id.clone();
    ui::with_menu(
        ui::with_preview(session_button(store, id, dated), move || {
            session_preview(store, preview)
        }),
        actions,
    )
}

/// The row's name as a field: Return or Save keeps it, Esc or Cancel
/// leaves it; an empty name goes back to the one taken from what was said.
fn rename_field(store: Store, id: String) -> View {
    let name = signal(untracked(|| store.title(&id)));
    let save = {
        let id = id.clone();
        move || store.rename_session(&id, name.get_untracked())
    };
    let keep = save.clone();
    view()
        .px_2()
        .py_1()
        .gap_1()
        .child(
            ui::field("Name", name)
                .label("Conversation name")
                .auto_focus(true)
                .on_submit(move |_: &str| keep())
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
                    move || store.set_renaming(None),
                )),
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

/// New conversation from the sidebar: inside a project's page it starts
/// there.
fn new_here(store: Store) {
    match untracked(|| store.sidebar_view()) {
        SidebarView::Project(id) => store.new_in_project(Some(id)),
        _ => store.select(None),
    }
}

/// How many projects Home shows before "All projects".
const HOME_PROJECTS: usize = 3;

/// Home: what needs the owner, what is working, the pinned, the projects
/// used last, then every other conversation by day.
fn home(
    store: Store,
    needs: Memo<Vec<String>>,
    working: Memo<Vec<String>>,
    rest: Memo<Vec<String>>,
) -> View {
    let pinned = computed(move || {
        rest.with(|ids| {
            ids.iter()
                .filter(|id| store.is_pinned(id))
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    let recent = computed(move || {
        rest.with(|ids| {
            ids.iter()
                .filter(|id| !store.is_pinned(id) && store.in_source(id))
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    let all = computed(move || store.projects());
    let shown = computed(move || {
        recent_projects(store)
            .into_iter()
            .take(HOME_PROJECTS)
            .collect::<Vec<_>>()
    });
    view()
        .gap(px(1.0))
        .child(group(store, "Needs you", needs))
        .child(group(store, "Working", working))
        .child(group(store, "Pinned", pinned))
        .child(section_label("Projects"))
        .child(when(
            move || store.project_form_open(),
            move || project_form(store),
        ))
        .child(each(
            shown,
            |id: &String| id.clone(),
            move |id, _| project_row(store, id.get_untracked(), false),
        ))
        .child(
            when(
                move || all.with(Vec::len) > HOME_PROJECTS,
                move || {
                    row(
                        || false,
                        move || store.show_sidebar_view(SidebarView::Projects),
                    )
                    .accessibility_label("All projects")
                    .child(icon(Glyph::Folder).text_color(colors::FAINT))
                    .child(
                        text("All projects")
                            .flex_1()
                            .text_size(px(ui::fit::body()))
                            .text_color(colors::MUTED),
                    )
                    .child(icon(Glyph::Chevron).text_color(colors::FAINT))
                },
            )
            .otherwise(move || new_project_row(store)),
        )
        // Where they came from, as one filter: only when they came from
        // more than one place.
        .child(when(
            move || !store.sources().is_empty(),
            move || {
                view()
                    .flex_row()
                    .items_center()
                    .child(section_label("Conversations").flex_1())
                    .child(source_filter(store))
            },
        ))
        .child(by_day(store, recent))
        .child(when(
            move || recent.with(Vec::is_empty) && store.source_filter() != Source::All,
            || {
                text("None from here.")
                    .px_2()
                    .py_1()
                    .text_size(px(type_scale::SM))
                    .text_color(colors::FAINT)
            },
        ))
}

/// The projects, the ones with the latest conversations first, then the
/// rest by name.
fn recent_projects(store: Store) -> Vec<String> {
    let mut projects = store.projects();
    projects.sort_by_key(|p| p.name.to_lowercase());
    let mut ids: Vec<String> = Vec::new();
    for session in store.session_ids() {
        if let Some(project) = store.session_project(&session)
            && !ids.contains(&project)
            && projects.iter().any(|p| p.id == project)
        {
            ids.push(project);
        }
    }
    for project in projects {
        if !ids.contains(&project.id) {
            ids.push(project.id);
        }
    }
    ids
}

/// A project's row: its name and how many conversations it has; a press
/// goes inside it. `detailed` adds its folder (the list of every project).
fn project_row(store: Store, id: String, detailed: bool) -> View {
    let (name, folder, label, open, count_id) =
        (id.clone(), id.clone(), id.clone(), id.clone(), id.clone());
    let count = computed(move || {
        store
            .session_ids()
            .iter()
            .filter(|s| {
                store.session_project(s).as_deref() == Some(count_id.as_str())
                    && !store.is_archived(s)
            })
            .count()
    });
    let (start, edit) = (id.clone(), id.clone());
    let actions = vec![
        ui::MenuAction::new("New conversation here", move || {
            store.new_in_project(Some(start.clone()))
        }),
        ui::MenuAction::new("Edit name or folder", move || {
            store.show_sidebar_view(SidebarView::Project(edit.clone()));
            store.edit_project(&edit);
        }),
    ];
    let mut names = view().flex_1().min_w(px(0.0)).child(
        text(move || store.project(&name).map(|p| p.name).unwrap_or_default())
            .number_of_lines(1)
            .text_size(px(ui::fit::body())),
    );
    if detailed {
        names = names.child(
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
        );
    }
    ui::with_menu(
        row(
            || false,
            move || store.show_sidebar_view(SidebarView::Project(open.clone())),
        )
        .accessibility_label(move || {
            Some(SharedString::from(format!(
                "Project {}, {} conversations",
                store.project(&label).map(|p| p.name).unwrap_or_default(),
                count.get()
            )))
        })
        .child(icon(Glyph::Folder).text_color(colors::MUTED))
        .child(names)
        .child(
            text(move || match count.get() {
                0 => String::new(),
                n => n.to_string(),
            })
            .tabular_nums()
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT),
        )
        .child(icon(Glyph::Chevron).text_color(colors::FAINT)),
        actions,
    )
}

/// "New project": opens the form.
fn new_project_row(store: Store) -> View {
    view().child(
        row(|| false, move || store.show_project_form(true))
            .accessibility_label("New project")
            .child(icon(Glyph::Plus).text_color(colors::FAINT))
            .child(
                text("New project")
                    .flex_1()
                    .text_size(px(ui::fit::body()))
                    .text_color(colors::MUTED),
            ),
    )
}

/// The top of a page inside the sidebar: Back, its title and what it
/// offers.
fn page_head(store: Store, title: impl Fn() -> String + 'static, tools: View) -> View {
    view()
        .flex_row()
        .items_center()
        .gap_1()
        .pt_1()
        .child(ui::icon_btn(
            Glyph::Back,
            "Back to all conversations",
            ui::Size::Md,
            move || store.show_sidebar_view(SidebarView::Home),
        ))
        .child(
            text(title)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(1)
                .role(AccessibilityRole::Header)
                .text_size(px(ui::fit::body()))
                .font_weight(FontWeight::SEMIBOLD),
        )
        .child(tools)
}

/// Every project, each with its folder, then New project.
fn all_projects(store: Store, _ids: Memo<Vec<String>>) -> View {
    let all = computed(move || recent_projects(store));
    view()
        .gap(px(1.0))
        .child(page_head(store, || "Projects".to_owned(), view()))
        .child(each(
            all,
            |id: &String| id.clone(),
            move |id, _| project_row(store, id.get_untracked(), true),
        ))
        .child(when(
            move || store.project_form_open(),
            move || project_form(store),
        ))
        .child(when(
            move || !store.project_form_open(),
            move || new_project_row(store),
        ))
}

/// One project: its name and folder, Edit, then its conversations by day
/// (New conversation, above, starts one here).
fn project_page(store: Store, id: String, ids: Memo<Vec<String>>) -> View {
    let mine = {
        let id = id.clone();
        computed(move || {
            ids.with(|ids| {
                ids.iter()
                    .filter(|s| store.session_project(s).as_deref() == Some(id.as_str()))
                    .cloned()
                    .collect::<Vec<_>>()
            })
        })
    };
    let (name, folder, edit, empty) = (id.clone(), id.clone(), id.clone(), id.clone());
    let tools = view().flex_row().items_center().child(ui::press_menu(
        Glyph::More,
        "Project options",
        vec![ui::MenuAction::new("Edit name or folder", move || {
            store.edit_project(&edit)
        })],
    ));
    view()
        .gap(px(1.0))
        .child(page_head(
            store,
            move || store.project(&name).map(|p| p.name).unwrap_or_default(),
            tools,
        ))
        .child(
            text(move || {
                store
                    .project(&folder)
                    .and_then(|p| p.path)
                    .unwrap_or_else(|| "No folder".to_owned())
            })
            .px_2()
            .pb_1()
            .number_of_lines(1)
            .text_size(px(type_scale::XS))
            .text_color(colors::FAINT),
        )
        .child(when(
            move || store.project_form_open(),
            move || project_form(store),
        ))
        .child(when(
            move || mine.with(Vec::is_empty),
            move || {
                let id = empty.clone();
                row(|| false, move || store.new_in_project(Some(id.clone()))).child(
                    text("No conversations yet. Start one")
                        .text_size(px(type_scale::SM))
                        .text_color(colors::FAINT),
                )
            },
        ))
        .child(by_day(store, mine))
}

/// Conversations under the day they were last active: Today, Yesterday,
/// Earlier (each with its date). A long day shows its first few, and Show
/// more brings the rest a few at a time.
fn by_day(store: Store, ids: Memo<Vec<String>>) -> View {
    let days_ago = move |id: &String| {
        store
            .last_activity(id)
            .map_or(i64::MAX, |at| ui::time::day(now_ms()) - ui::time::day(at))
    };
    let part = move |keep: fn(i64) -> bool| {
        computed(move || {
            ids.with(|ids| {
                ids.iter()
                    .filter(|id| keep(days_ago(id)))
                    .cloned()
                    .collect::<Vec<_>>()
            })
        })
    };
    view()
        .gap(px(1.0))
        .child(day_group(store, "Today", part(|d| d <= 0), false))
        .child(day_group(store, "Yesterday", part(|d| d == 1), false))
        .child(day_group(store, "Earlier", part(|d| d > 1), true))
}

/// How many of a day's conversations show at first, and how many more each
/// Show more brings.
const DAY_FIRST: usize = 5;
const DAY_MORE: usize = 10;

/// One day's conversations, the first [`DAY_FIRST`] of them, then Show
/// more (and Show less once more show).
fn day_group(store: Store, title: &'static str, ids: Memo<Vec<String>>, dated: bool) -> View {
    let limit = signal(DAY_FIRST);
    let shown = computed(move || {
        let limit = limit.get();
        ids.with(|ids| ids.iter().take(limit).cloned().collect::<Vec<_>>())
    });
    let hidden = move || ids.with(Vec::len).saturating_sub(limit.get());
    view()
        .gap(px(1.0))
        .child(when(
            move || !ids.with(Vec::is_empty),
            move || section_label(title),
        ))
        .child(each(
            shown,
            |id: &String| id.clone(),
            move |id, _| session_row(store, id.get_untracked(), dated),
        ))
        .child(when(
            move || hidden() > 0 || limit.get() > DAY_FIRST,
            move || {
                view()
                    .flex_row()
                    .items_center()
                    .child(when(
                        move || hidden() > 0,
                        move || {
                            more_row(
                                move || format!("Show {} more", hidden().min(DAY_MORE)),
                                move || limit.update(|l| *l += DAY_MORE),
                            )
                        },
                    ))
                    .child(when(
                        move || limit.get() > DAY_FIRST,
                        move || more_row(|| "Show less".to_owned(), move || limit.set(DAY_FIRST)),
                    ))
            },
        ))
}

/// A quiet press under a list: Show more, Show less.
fn more_row(label: impl Fn() -> String + Clone + 'static, on_press: impl Fn() + 'static) -> View {
    let name = label.clone();
    view().child(
        row(|| false, on_press)
            .accessibility_label(move || Some(SharedString::from(name())))
            // Lined up with the titles, past the dot's slot.
            .child(view().w(px(ui::fit::icon_slot())))
            .child(
                text(label)
                    .text_size(px(type_scale::SM))
                    .text_color(colors::MUTED),
            ),
    )
}

/// Where the listed conversations came from, as one filter.
fn source_filter(store: Store) -> View {
    let shown = move || store.source_filter().label().to_owned();
    ui::pick_list(
        ui::Picker {
            title: "Show conversations from",
            items: move || {
                let chosen = store.source_filter();
                store
                    .sources()
                    .into_iter()
                    .enumerate()
                    .map(|(at, source)| {
                        ui::PickItem::new(at.to_string(), source.label()).checked(source == chosen)
                    })
                    .collect()
            },
            pick: move |picked: String| {
                if let Some(source) = picked
                    .parse::<usize>()
                    .ok()
                    .and_then(|at| store.sources().into_iter().nth(at))
                {
                    store.set_source_filter(source);
                }
            },
            action: None,
        },
        move || format!("Show conversations from: {}", shown()),
        view()
            .min_h(px(ui::fit::row()))
            .flex_row()
            .items_center()
            .gap_1()
            .px_2()
            .child(
                text(shown)
                    .text_size(px(ui::fit::small()))
                    .text_color(colors::MUTED),
            )
            .child(
                icon(Glyph::Down)
                    .text_color(colors::FAINT)
                    .accessibility_elements_hidden(true),
            ),
        false,
    )
}

/// The project form: a new project, or the one being edited.
fn project_form(store: Store) -> View {
    let editing = untracked(|| store.project_editing().and_then(|id| store.project(&id)));
    let name = signal(editing.as_ref().map(|p| p.name.clone()).unwrap_or_default());
    let path = signal(
        editing
            .as_ref()
            .and_then(|p| p.path.clone())
            .unwrap_or_default(),
    );
    let (saving_label, folder_label) = if editing.is_some() {
        ("Save", "Make the folder and save")
    } else {
        ("Create project", "Make the folder and the project")
    };
    // The daemon refused only because the folder is missing, and the field
    // still names that folder: offer to make it.
    let can_make_folder = move || {
        store
            .project_missing_folder()
            .is_some_and(|missing| path.with(|typed| typed.trim() == missing))
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
        // A missing folder the daemon can make is a question, not an error.
        .child(when(
            move || store.project_error().is_some() && !can_make_folder(),
            move || {
                ui::banner(
                    move || store.project_error().unwrap_or_default(),
                    Tone::Danger,
                )
            },
        ))
        .child(when(can_make_folder, move || {
            ui::banner(
                move || {
                    format!(
                        "{} doesn't exist yet. Make it?",
                        store.project_missing_folder().unwrap_or_default()
                    )
                },
                Tone::Neutral,
            )
        }))
        .child(
            view()
                .flex_row()
                .gap_2()
                .child(when(
                    move || !can_make_folder(),
                    move || {
                        ui::btn(
                            saving_label,
                            ui::Variant::Primary,
                            ui::Size::Sm,
                            move || {
                                store.save_project(
                                    &name.get_untracked(),
                                    &path.get_untracked(),
                                    false,
                                )
                            },
                        )
                    },
                ))
                .child(when(can_make_folder, move || {
                    ui::btn(
                        folder_label,
                        ui::Variant::Primary,
                        ui::Size::Sm,
                        move || {
                            store.save_project(&name.get_untracked(), &path.get_untracked(), true)
                        },
                    )
                }))
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
/// The project picker of a conversation's "Move to project…".
pub fn move_picker(store: Store) -> View {
    ui::pick_dialog(
        ui::Picker {
            title: "Move to project",
            items: move || {
                let current = store.moving().and_then(|id| store.session_project(&id));
                project_choices(store, current, Some("No project"))
            },
            pick: move |project: String| {
                if let Some(session) = store.moving() {
                    store.move_to_project(&session, Some(project).filter(|p| !p.is_empty()));
                }
            },
            action: None,
        },
        store.moving_open(),
    )
}

/// The projects to choose from, `chosen` checked: past four, the three
/// last worked in under Recent, then the rest by name under All projects.
/// `none` (a title) adds a first choice of no project, id "".
pub fn project_choices(
    store: Store,
    chosen: Option<String>,
    none: Option<&str>,
) -> Vec<ui::PickItem> {
    let mut projects = store.projects();
    projects.sort_by_key(|p| p.name.to_lowercase());
    let item = |p: &aidaemon_client_protocol::ProjectSummary, group: Option<&str>| {
        let row = ui::PickItem::new(p.id.clone(), p.name.clone())
            .detail(
                p.path
                    .as_deref()
                    .map_or_else(|| "No folder".to_owned(), short_path),
            )
            .glyph(ui::Glyph::Folder)
            .checked(chosen.as_deref() == Some(p.id.as_str()));
        match group {
            Some(group) => row.group(group),
            None => row,
        }
    };
    let mut out = Vec::new();
    if let Some(title) = none {
        out.push(ui::PickItem::new("", title).checked(chosen.is_none()));
    }
    if projects.len() <= 4 {
        out.extend(projects.iter().map(|p| item(p, None)));
        return out;
    }
    let mut recent = Vec::<String>::new();
    for id in store.session_ids() {
        if let Some(project) = store.session_project(&id)
            && !recent.contains(&project)
        {
            recent.push(project);
            if recent.len() == 3 {
                break;
            }
        }
    }
    out.extend(
        recent
            .iter()
            .filter_map(|id| projects.iter().find(|p| &p.id == id))
            .map(|p| item(p, Some("Recent"))),
    );
    out.extend(
        projects
            .iter()
            .filter(|p| !recent.contains(&p.id))
            .map(|p| item(p, Some("All projects"))),
    );
    out
}

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

fn session_button(store: Store, id: String, dated: bool) -> nsh_base::ButtonElement {
    let (a, b, c, d, e) = (id.clone(), id.clone(), id.clone(), id.clone(), id.clone());
    let (f, g, h) = (id.clone(), id.clone(), id.clone());
    let label_id = id.clone();
    let date_id = id.clone();
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
    // Working (accent, pulsing) or a new reply (filled); nothing while
    // quiet, in the same slot so the names line up.
    .child(ui::state_dot(
        move || store.is_running(&c) || store.is_unread(&c),
        move || {
            if store.is_running(&e) {
                colors::ACCENT
            } else if store.is_unread(&e) {
                colors::TEXT
            } else {
                colors::TRANSPARENT
            }
        },
    ))
    .child(
        text(move || store.title(&d))
            .flex_1()
            .min_w(px(0.0))
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
    // One faint word where it adds something: its project (outside that
    // project's page), or the chat it came from when not the app.
    .child(
        text(move || {
            if let Some(project) = store.session_project(&g).and_then(|p| store.project(&p)) {
                return match store.sidebar_view() {
                    SidebarView::Project(_) => String::new(),
                    _ => project.name,
                };
            }
            store.source_tag(&id)
        })
        .max_w(percentage(0.4))
        .number_of_lines(1)
        .text_size(px(if ui::touch() { 12.0 } else { type_scale::XS }))
        .text_color(colors::FAINT),
    )
    // Under Earlier: the day it was last active ("Oct 30").
    .child(when(dated, move || {
        let at = date_id.clone();
        text(move || {
            store
                .last_activity(&at)
                .map(|at| ui::time::short_date(at, now_ms()))
                .unwrap_or_default()
        })
        .tabular_nums()
        .text_size(px(if ui::touch() { 12.0 } else { type_scale::XS }))
        .text_color(colors::FAINT)
    }))
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
                .child(text("Automations").flex_1().text_size(px(ui::fit::body())))
                // Runs with something to read since the last look.
                .child(when(
                    move || store.automation_news() > 0,
                    move || {
                        ui::badge(
                            move || format!("{} new", store.automation_news()),
                            Tone::Accent,
                        )
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
            move |id, _| session_row(store, id.get_untracked(), false),
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
            // Conversations already listed above match: "nothing" here
            // would read as if the search found nothing at all.
            SearchRead::Ready { hits, .. }
                if hits.is_empty() && !store.visible_ids().is_empty() =>
            {
                view()
            }
            SearchRead::Ready { hits, .. } => {
                let mut list = view().gap(px(1.0)).child(section_label(if hits.is_empty() {
                    "Nothing matches, in titles or messages"
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

/// At the end of the list: the archived conversations a press away, and
/// the way back. Nothing while none is archived.
fn archived_toggle(store: Store) -> View {
    view().pt_2().child(dynamic(move || {
        let count = store.archived_count();
        let showing = store.showing_archived();
        Some(untracked(move || {
            if count == 0 && !showing {
                return view();
            }
            let label = if showing {
                "Back to your conversations".to_string()
            } else {
                format!("Archived · {count}")
            };
            view().child(
                row(|| false, move || store.show_archived(!showing))
                    .accessibility_label(label.clone())
                    .child(icon(Glyph::Inbox).text_color(colors::FAINT))
                    .child(
                        text(label)
                            .flex_1()
                            .text_size(px(type_scale::SM))
                            .text_color(colors::MUTED),
                    ),
            )
        }))
    }))
}
