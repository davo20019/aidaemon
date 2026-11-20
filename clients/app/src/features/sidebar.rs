//! The sidebar. Home: its name, Search (a field once pressed) and one New
//! conversation, the conversations
//! that need the owner or are working, the pinned ones, the projects used
//! last (one line each; a press goes inside one), then every conversation
//! newest first, each with its time and tagged with its project or the
//! chat it came from. A project's
//! page: its folder, its conversations, New conversation there, Edit.
//! Then the pages and, when something is wrong, the connection.

use nsh_core::prelude::*;

use crate::store::{Felt, Page, SearchRead, SidebarView, Source, Store, now_ms};
use crate::ui::{self, Glyph, Tone, colors, icon, row, section_label, status, type_scale};
use aidaemon_client_core::machine::DropKind;

pub fn sidebar(store: Store) -> View {
    let ids = computed(move || store.visible_ids());
    // What waits on the owner first; the rest by time, each row saying
    // whether it works.
    let needs = computed(move || {
        ids.with(|ids| {
            ids.iter()
                .filter(|id| store.needs_you(id))
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    let rest = computed(move || {
        ids.with(|ids| {
            ids.iter()
                .filter(|id| !store.needs_you(id))
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    let query = signal(store.filter());
    // The dot by the name, pressed: its words at the foot until pressed again.
    let status_shown = signal(false);
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
        // The top line: the name, Search, (a phone's) New conversation and
        // Close; Search turns the line into the field, Cancel turns it back.
        .child(top_line(store, query, status_shown))
        .child(when(!ui::touch(), move || {
            view().px_2().child(
                row(
                    move || store.page() == Page::Conversation && store.selected().is_none(),
                    move || new_here(store),
                )
                .accessibility_label("New conversation")
                .child(icon(Glyph::Compose).text_color(colors::MUTED))
                .child(
                    text(move || match store.sidebar_view() {
                        SidebarView::Project(id) => format!(
                            "New conversation in {}",
                            store.project(&id).map(|p| p.name).unwrap_or_default()
                        ),
                        _ => "New conversation".to_owned(),
                    })
                    .flex_1()
                    .number_of_lines(1)
                    .text_size(px(type_scale::BASE)),
                )
                .child(ui::kbd(key_label("mod+n"))),
            )
        }))
        .child(ui::refreshable(
            scroll().flex_1().child(
                view()
                    .px_2()
                    .pb_2()
                    .gap(px(1.0))
                    .child(dynamic(move || {
                        // Searching lists every match flat, and so do the
                        // archived; otherwise the view the owner is in.
                        let flat = searching();
                        let archived = store.showing_archived();
                        let shown = store.sidebar_view();
                        Some(untracked(move || {
                            if archived && !flat {
                                // A project's own, asked for on its page.
                                let project = match &shown {
                                    SidebarView::Project(id) => Some(id.clone()),
                                    _ => None,
                                };
                                return archived_page(store, project, ids);
                            }
                            if flat {
                                return view()
                                    .gap(px(1.0))
                                    .child(group(store, "Needs you", needs))
                                    .child(group(store, "Conversations", rest));
                            }
                            match shown {
                                SidebarView::Home => home(store, needs, rest),
                                SidebarView::Projects => all_projects(store, ids),
                                SidebarView::Project(id) => project_page(store, id, ids),
                            }
                        }))
                    }))
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
            move || store.pull_refreshing(),
            move || store.pull_refresh(),
        ))
        .child(pages(store))
        // The line shows only when something is wrong: the dot by the name
        // says connected or reconnecting.
        .child(
            view()
                .style_when(
                    move || {
                        !status_shown.get()
                            && matches!(
                                store.felt(),
                                Felt::Online | Felt::Connecting | Felt::Reconnecting
                            )
                    },
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
    let (preview, title) = (id.clone(), id.clone());
    // While the menu is up its row is the marked one, whichever
    // conversation is open behind it.
    let about = signal(false);
    ui::with_menu_about(
        move || store.title(&title),
        about,
        ui::with_preview(
            session_button(store, id, dated).style_when(move || about.get(), ui::menu_target),
            move || session_preview(store, preview),
        ),
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

/// The sidebar's top line. Shut: "AIDaemon", Search, New conversation
/// (a phone's; a window has its own row) and Close (or ⌘K's hint). Open:
/// the search field across the line, focused, and Cancel, which clears it.
/// A search left in the field keeps it open.
/// The name button's own padding, taken from the line's so the name stays
/// where it was.
const NAME_INSET: f32 = 4.0;

fn top_line(store: Store, query: Signal<String>, status_shown: Signal<bool>) -> View {
    let open = signal(!untracked(|| query.get()).trim().is_empty());
    let close = move || {
        query.set(String::new());
        open.set(false);
    };
    view()
        .h(px(ui::fit::bar()))
        .flex_row()
        .items_center()
        .pl(px(12.0 - NAME_INSET))
        .pr_2()
        .gap_1()
        .child(
            when(
                move || open.get(),
                move || {
                    view()
                        .flex_1()
                        .flex_row()
                        .items_center()
                        .gap_1()
                        .child(
                            ui::field("Search", query)
                                .label("Search conversations")
                                .auto_focus(true)
                                .flex_1(),
                        )
                        .child(ui::btn("Cancel", ui::Variant::Ghost, ui::Size::Sm, close))
                },
            )
            .otherwise(move || {
                view()
                    .flex_1()
                    .flex_row()
                    .items_center()
                    .gap_1()
                    // The name goes home: the main list and a new
                    // conversation.
                    .child(ui::name_btn("AIDaemon", NAME_INSET, move || {
                        store.show_sidebar_view(SidebarView::Home);
                        store.select(None);
                    }))
                    // The connection at a glance; its words come to the
                    // footer only when something is wrong.
                    .child(ui::status_dot(
                        move || felt_status(&store.felt()).0,
                        move || felt_status(&store.felt()).1,
                        move || status_shown.update(|shown| *shown = !*shown),
                    ))
                    .child(view().flex_1())
                    .child(ui::icon_btn(
                        Glyph::Search,
                        "Search conversations",
                        ui::Size::Md,
                        move || open.set(true),
                    ))
                    .child(when(ui::touch(), move || {
                        ui::icon_btn(
                            Glyph::Compose,
                            "New conversation",
                            ui::Size::Md,
                            move || new_here(store),
                        )
                    }))
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
                    )
            }),
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

/// Home lists every project up to this many; past it, the pinned and the
/// ones used last (one fewer) and "All projects" in the last one's place.
const HOME_PROJECTS: usize = 4;

/// Home: what needs the owner, the pinned, the projects
/// used last, then every other conversation, newest first.
fn home(store: Store, needs: Memo<Vec<String>>, rest: Memo<Vec<String>>) -> View {
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
    // As sorted (the pinned first, then the owner's order, else the ones
    // used last); "All projects" takes the last place when some are left
    // out.
    let shown = computed(move || {
        let room = if all.with(Vec::len) > HOME_PROJECTS {
            HOME_PROJECTS - 1
        } else {
            HOME_PROJECTS
        };
        store.home_projects(room)
    });
    let more = computed(move || all.with(Vec::len) > shown.with(Vec::len));
    let reorder = Reorder::new(shown, false);
    view()
        .gap(px(1.0))
        .child(group(store, "Needs you", needs))
        .child(group(store, "Pinned", pinned))
        .child(section_label("Projects"))
        .child(when(
            move || store.project_form_open(),
            move || project_form(store),
        ))
        .child(view().gap(px(ROW_GAP)).child(each(
            shown,
            |id: &String| id.clone(),
            move |id, _| project_row(store, id.get_untracked(), Some(reorder)),
        )))
        .child(
            when(more, move || {
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
            })
            .otherwise(move || new_project_row(store)),
        )
        // Their heading, with which ones are listed as one filter: where
        // they came from (when from more than one place) and the archived
        // (when there are any).
        .child(
            view()
                .flex_row()
                .items_center()
                .child(section_label("Recent").flex_1())
                .child(when(
                    move || !store.sources().is_empty() || store.archived_count() > 0,
                    move || source_filter(store),
                )),
        )
        .child(listed(store, recent))
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

/// A drag reordering a list of projects (Home's, or every project's):
/// which one and how far it moved, the list as it stands, and how tall its
/// rows are (`detailed`: a second line, the folder).
#[derive(Clone, Copy)]
struct Reorder {
    drag: Signal<Option<(String, f32)>>,
    order: Memo<Vec<String>>,
    detailed: bool,
}

/// The gap between two project rows.
const ROW_GAP: f32 = 1.0;

impl Reorder {
    fn new(order: Memo<Vec<String>>, detailed: bool) -> Self {
        Self {
            drag: signal(None),
            order,
            detailed,
        }
    }

    /// A row's height.
    fn row(&self) -> f32 {
        ui::fit::row() + if self.detailed { 14.0 } else { 0.0 }
    }

    /// A drag moves a row a place each this far.
    fn step(&self) -> f32 {
        self.row() + ROW_GAP
    }
}

impl Reorder {
    /// Where the dragged row would land: `(from, to)`.
    fn target(&self) -> Option<(usize, usize)> {
        let (id, dy) = self.drag.get()?;
        self.order.with(|ids| {
            let from = ids.iter().position(|p| *p == id)?;
            let step = self.step();
            let to = (from as f32 + (dy / step).round()).clamp(0.0, (ids.len() - 1) as f32);
            Some((from, to as usize))
        })
    }

    /// How far row `id` moves while another is dragged: a place, out of
    /// the dragged one's way; the dragged one follows the pointer.
    fn offset(&self, id: &str) -> f32 {
        let Some((dragged, dy)) = self.drag.get() else {
            return 0.0;
        };
        if dragged == id {
            return dy;
        }
        let Some((from, to)) = self.target() else {
            return 0.0;
        };
        let at = self.order.with(|ids| ids.iter().position(|p| p == id));
        let step = self.step();
        match at {
            Some(at) if from < to && at > from && at <= to => -step,
            Some(at) if to < from && at >= to && at < from => step,
            _ => 0.0,
        }
    }
}

/// A project's row: its name and how many conversations it has; a press
/// goes inside it, its menu pins, moves or deletes it. `reorder` adds a
/// handle to drag it by (and, detailed, its folder).
fn project_row(store: Store, id: String, reorder: Option<Reorder>) -> View {
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
    let (start, edit, pin, gone) = (id.clone(), id.clone(), id.clone(), id.clone());
    let (up, can_up, down, can_down) = (id.clone(), id.clone(), id.clone(), id.clone());
    let pinned_id = id.clone();
    let pinned = move || store.is_project_pinned(&pinned_id);
    let pin_label = if untracked(&pinned) {
        "Unpin"
    } else {
        "Pin to the top"
    };
    let actions = vec![
        ui::MenuAction::new("New conversation here", move || {
            store.new_in_project(Some(start.clone()))
        }),
        ui::MenuAction::new(pin_label, move || store.toggle_project_pin(&pin)),
        ui::MenuAction::new("Move up", move || store.move_project_by(&up, -1))
            .shown_when(move || reorder.is_some() && store.can_move_project(&can_up, -1)),
        ui::MenuAction::new("Move down", move || store.move_project_by(&down, 1))
            .shown_when(move || reorder.is_some() && store.can_move_project(&can_down, 1)),
        ui::MenuAction::new("Edit name or folder", move || {
            store.show_sidebar_view(SidebarView::Project(edit.clone()));
            store.edit_project(&edit);
        }),
        ui::MenuAction::new("Delete project…", move || confirm_delete(store, &gone)),
    ];
    let mut names = view().flex_1().min_w(px(0.0)).child(
        text(move || store.project(&name).map(|p| p.name).unwrap_or_default())
            .number_of_lines(1)
            .text_size(px(ui::fit::body())),
    );
    if reorder.is_some_and(|r| r.detailed) {
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
    let star = pinned.clone();
    let mut line = row(
        || false,
        move || store.show_sidebar_view(SidebarView::Project(open.clone())),
    )
    .accessibility_label(move || {
        Some(SharedString::from(format!(
            "Project {}{}, {} conversations",
            store.project(&label).map(|p| p.name).unwrap_or_default(),
            if pinned() { ", pinned" } else { "" },
            count.get()
        )))
    })
    .child(icon(Glyph::Folder).text_color(colors::MUTED))
    .child(names)
    .child(when(star, || icon(Glyph::Pinned).text_color(colors::FAINT)))
    .child(
        text(move || match count.get() {
            0 => String::new(),
            n => n.to_string(),
        })
        .tabular_nums()
        .text_size(px(type_scale::XS))
        .text_color(colors::FAINT),
    );
    let about = signal(false);
    line = line.style_when(move || about.get(), ui::menu_target);
    let about_id = id.clone();
    let about_name = move || store.project(&about_id).map(|p| p.name).unwrap_or_default();
    let Some(reorder) = reorder else {
        return ui::with_menu_about(
            about_name,
            about,
            line.child(icon(Glyph::Chevron).text_color(colors::FAINT)),
            actions,
        );
    };
    line = line.h(px(reorder.row()));
    let (moving, offset_id, handle_id) = (id.clone(), id.clone(), id.clone());
    let lifted = move || {
        reorder
            .drag
            .with(|d| d.as_ref().is_some_and(|(d, _)| *d == moving))
    };
    view()
        .translate_y(move || DefiniteLength::from(px(reorder.offset(&offset_id))))
        .z_index({
            let lifted = lifted.clone();
            move || if lifted() { 1 } else { 0 }
        })
        .rounded(px(ui::radius::CONTROL))
        .style_when(lifted, |s| s.bg(colors::SURFACE).shadow_md())
        .child(
            view()
                .flex_row()
                .items_center()
                .child(
                    view()
                        .flex_1()
                        .min_w(px(0.0))
                        .child(ui::with_menu_about(about_name, about, line, actions)),
                )
                .child(drag_handle(store, reorder, handle_id)),
        )
}

/// The handle a project's row is dragged by, up or down the list.
fn drag_handle(store: Store, reorder: Reorder, id: String) -> View {
    let (grant, end) = (id.clone(), id.clone());
    view()
        .test_id(&format!("drag:{id}"))
        .w(px(ui::fit::icon_slot() + 8.0))
        .h(px(reorder.row()))
        .items_center()
        .justify_center()
        // The mouse's and a finger's; the keyboard and a screen reader
        // have Move up and Move down in the row's menu.
        .child(icon(Glyph::Grip).text_color(colors::FAINT))
        .claim_touches(TouchClaim::on_start())
        .keep_touches(true)
        .on_responder_grant(move |_: &ResponderEvent| {
            reorder.drag.set(Some((grant.clone(), 0.0)));
        })
        .on_responder_move(move |e: &ResponderEvent| {
            reorder.drag.update(|d| {
                if let Some((_, dy)) = d {
                    *dy = e.gesture.delta.y.0;
                }
            });
        })
        .on_responder_release(move |_: &ResponderEvent| {
            let target = untracked(|| reorder.target());
            reorder.drag.set(None);
            if let Some((from, to)) = target
                && from != to
            {
                store.move_project(&end, to);
            }
        })
        .on_responder_terminate(move |_: &ResponderEvent| reorder.drag.set(None))
}

/// Ask, then delete a project: its conversations go to no project, its
/// folder stays.
fn confirm_delete(store: Store, id: &str) {
    let Some(project) = untracked(|| store.project(id)) else {
        return;
    };
    let chats = untracked(|| {
        store
            .session_ids()
            .iter()
            .filter(|s| store.session_project(s).as_deref() == Some(id))
            .count()
    });
    let moved = match chats {
        0 => "It has no conversations.".to_owned(),
        1 => "Its conversation moves to No project.".to_owned(),
        n => format!("Its {n} conversations move to No project."),
    };
    let folder = match project.path {
        Some(path) => format!(" The folder {} stays as it is.", short_path(&path)),
        None => String::new(),
    };
    let id = id.to_owned();
    ui::confirm(
        format!("Delete {}?", project.name),
        format!("{moved}{folder}"),
        "Delete",
        move || store.delete_project(&id),
    );
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
    let in_project = untracked(|| {
        store.showing_archived() && matches!(store.sidebar_view(), SidebarView::Project(_))
    });
    view()
        .flex_row()
        .items_center()
        .gap_1()
        .pt_1()
        .child(ui::icon_btn(
            Glyph::Back,
            if in_project {
                "Back to the project"
            } else {
                "Back to all conversations"
            },
            ui::Size::Md,
            move || {
                // A project's archived are a page of the project's.
                if in_project {
                    store.show_archived(false);
                } else {
                    store.show_sidebar_view(SidebarView::Home)
                }
            },
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

/// Every project, each with its folder, the pinned first and each dragged
/// into place by its handle, then New project.
fn all_projects(store: Store, _ids: Memo<Vec<String>>) -> View {
    let reorder = Reorder::new(computed(move || store.sorted_projects()), true);
    view()
        .gap(px(1.0))
        .child(page_head(store, || "Projects".to_owned(), view()))
        .child(view().gap(px(ROW_GAP)).child(each(
            reorder.order,
            |id: &String| id.clone(),
            move |id, _| project_row(store, id.get_untracked(), Some(reorder)),
        )))
        .child(when(
            move || store.project_form_open(),
            move || project_form(store),
        ))
        .child(when(
            move || !store.project_form_open(),
            move || new_project_row(store),
        ))
}

/// One project: its name and folder, Edit, then its conversations
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
    let put_away = id.clone();
    let (browse, browsing, has_folder) = (id.clone(), id.clone(), id.clone());
    let (pin, unpin, pinned, unpinned, gone) =
        (id.clone(), id.clone(), id.clone(), id.clone(), id.clone());
    // Its archived, an icon beside the menu (only when it has some).
    let archived = dynamic(move || {
        let any = store.session_ids().iter().any(|s| {
            store.is_archived(s) && store.session_project(s).as_deref() == Some(put_away.as_str())
        });
        any.then(|| {
            untracked(|| {
                ui::icon_btn(
                    Glyph::Archive,
                    "Archived conversations in this project",
                    ui::Size::Md,
                    move || store.show_archived(true),
                )
            })
        })
    });
    let tools = view()
        .flex_row()
        .items_center()
        .child(archived)
        .child(ui::press_menu(
            Glyph::More,
            "Project options",
            vec![
                ui::MenuAction::new("Pin to the top", move || store.toggle_project_pin(&pin))
                    .shown_when(move || !store.is_project_pinned(&unpinned)),
                ui::MenuAction::new("Unpin", move || store.toggle_project_pin(&unpin))
                    .shown_when(move || store.is_project_pinned(&pinned)),
                ui::MenuAction::new("Edit name or folder", move || store.edit_project(&edit)),
                ui::MenuAction::new("Delete project…", move || confirm_delete(store, &gone)),
            ],
        ));
    view()
        .gap(px(1.0))
        .child(page_head(
            store,
            move || store.project(&name).map(|p| p.name).unwrap_or_default(),
            tools,
        ))
        // Its folder, the way into its files (a page of their own, no
        // conversation needed); without one, the way to add one.
        .child(dynamic(move || {
            let path = store
                .project(&folder)
                .and_then(|p| p.path)
                .filter(|f| !f.trim().is_empty());
            let (open, on, add) = (browse.clone(), browsing.clone(), has_folder.clone());
            Some(untracked(move || match path {
                Some(path) => row(
                    move || store.showing_files_of(&on),
                    move || store.open_project_files(&open),
                )
                .accessibility_label("This project's files")
                .child(icon(Glyph::Folder).text_color(colors::MUTED))
                .child(
                    text(short_path(&path))
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .font_family(ui::theme::MONO)
                        .text_size(px(type_scale::SM))
                        .text_color(colors::MUTED),
                )
                .child(icon(Glyph::Chevron).text_color(colors::FAINT)),
                None => row(|| false, move || store.edit_project(&add))
                    .accessibility_label("Add a folder to this project")
                    .child(icon(Glyph::Folder).text_color(colors::FAINT))
                    .child(
                        text("Add a folder…")
                            .flex_1()
                            .number_of_lines(1)
                            .text_size(px(type_scale::SM))
                            .text_color(colors::MUTED),
                    ),
            }))
        }))
        .child(when(
            move || store.project_form_open(),
            move || project_form(store),
        ))
        // On a phone, starting one here is in view, not only behind the
        // pencil (a window's top row already names the project): it opens
        // a new Code conversation in this project's folder.
        .child(when(ui::touch(), move || {
            let start = empty.clone();
            row(|| false, move || store.new_in_project(Some(start.clone())))
                .accessibility_label("New conversation in this project")
                .child(icon(Glyph::Compose).text_color(colors::MUTED))
                .child(
                    text("New conversation")
                        .flex_1()
                        .number_of_lines(1)
                        .text_size(px(type_scale::BASE)),
                )
        }))
        .child(when(
            move || mine.with(Vec::is_empty),
            move || {
                text("No conversations yet")
                    .px_2()
                    .py_1()
                    .text_size(px(type_scale::SM))
                    .text_color(colors::FAINT)
            },
        ))
        .child(listed(store, mine))
}

/// Conversations newest first, each with when it last moved; no day
/// headings, nothing held back.
fn listed(store: Store, ids: Memo<Vec<String>>) -> View {
    view().gap(px(1.0)).child(each(
        ids,
        |id: &String| id.clone(),
        move |id, _| session_row(store, id.get_untracked(), true),
    ))
}

/// Which conversations are listed, as one filter: where they came from,
/// and last the archived, which open as their own page.
fn source_filter(store: Store) -> View {
    const ARCHIVED: &str = "archived";
    let shown = move || store.source_filter().label().to_owned();
    // With one place only there is still "All" to be on.
    let sources = move || match store.sources() {
        found if found.is_empty() => vec![Source::All],
        found => found,
    };
    ui::pick_list(
        ui::Picker {
            title: "Show",
            items: move || {
                let chosen = store.source_filter();
                let mut items = sources()
                    .into_iter()
                    .enumerate()
                    .map(|(at, source)| {
                        let item = ui::PickItem::new(at.to_string(), source.label())
                            .checked(source == chosen);
                        match filter_glyph(source) {
                            Some(glyph) => item.glyph(glyph),
                            None => item,
                        }
                    })
                    .collect::<Vec<_>>();
                match store.archived_count() {
                    0 => {}
                    count => items.push(
                        ui::PickItem::new(ARCHIVED, "Archived")
                            .detail(if count == 1 {
                                "1 conversation".to_owned()
                            } else {
                                format!("{count} conversations")
                            })
                            .glyph(Glyph::Archive),
                    ),
                }
                items
            },
            pick: move |picked: String| {
                if picked == ARCHIVED {
                    store.show_archived(true);
                } else if let Some(source) = picked
                    .parse::<usize>()
                    .ok()
                    .and_then(|at| sources().into_iter().nth(at))
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
            .child(dynamic(move || {
                let glyph = filter_glyph(store.source_filter())?;
                Some(untracked(move || {
                    icon(glyph)
                        .text_color(colors::MUTED)
                        .accessibility_elements_hidden(true)
                }))
            }))
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
        // Lit only while the conversation is what the page shows, not
        // behind another page (Files, Memory, …).
        move || {
            store.page() == Page::Conversation && store.selected().as_deref() == Some(a.as_str())
        },
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
    // A new reply: a filled dot; nothing otherwise, in the same slot so
    // the names line up.
    .child(ui::state_dot(
        move || store.is_unread(&c),
        move || {
            if store.is_unread(&e) {
                colors::TEXT
            } else {
                colors::TRANSPARENT
            }
        },
    ))
    .child(row_names(store, id, d, f, g, h))
    // At the end, what it is doing: waiting on the owner ("Needs you",
    // amber), working (a turning ring), else when it last moved ("3h",
    // "Oct 30") where the list shows times.
    .child(dynamic(move || {
        let state = if store.needs_you(&date_id) {
            RowState::NeedsYou
        } else if store.is_working(&date_id) {
            // Offline, the app only knows it was working when the link
            // dropped: the finish may have happened since.
            if matches!(store.felt(), Felt::Online) {
                RowState::Working
            } else {
                RowState::WasWorking
            }
        } else {
            RowState::Quiet
        };
        let at = date_id.clone();
        Some(untracked(move || match state {
            RowState::NeedsYou => ui::badge_dot("Needs you", Tone::Warning),
            RowState::Working => ui::spinner("Working"),
            RowState::WasWorking => ui::spinner_stale("Was working when the connection dropped"),
            RowState::Quiet => view().child(when(dated, move || {
                let at = at.clone();
                text(move || {
                    store
                        .last_activity(&at)
                        .map(|at| ui::time::stamp(at, now_ms()))
                        .unwrap_or_default()
                })
                .tabular_nums()
                .text_size(px(if ui::touch() { 12.0 } else { type_scale::XS }))
                .text_color(colors::FAINT)
            })),
        }))
    }))
}

/// A conversation row's name and where it is from: its project's name
/// (outside that project's page), or the chat it came from when not the
/// app (Telegram, Discord and Slack by their icons, a group adding people;
/// any other by name). A window keeps them on one line, faint after the
/// name; a phone, whose name has little room, puts them on a second line
/// under it, and a row with nothing to say stays one line.
fn row_names(store: Store, id: String, d: String, f: String, g: String, h: String) -> View {
    let title = text(move || store.title(&d))
        .flex_1()
        .min_w(px(0.0))
        .number_of_lines(1)
        .text_size(px(ui::fit::body()))
        // The name is what a row is found by: body text always, a new
        // reply adding weight (beside its dot) rather than brightness.
        .text_color(colors::TEXT)
        .style_when(
            move || store.is_unread(&h),
            |s| s.font_weight(FontWeight::MEDIUM),
        );
    let star = text(move || if store.is_pinned(&f) { "★" } else { "" })
        .text_size(px(type_scale::XS))
        .text_color(colors::FAINT);
    let (tag_id, mark_id, any_id) = (g, id.clone(), id);
    let tag = move || row_tag(store, &tag_id);
    let tag_text = move || {
        let tag = tag.clone();
        text(tag)
            .number_of_lines(1)
            .text_size(px(if ui::touch() { 12.0 } else { type_scale::XS }))
            .text_color(colors::MUTED)
    };
    let mark = move || {
        let id = mark_id.clone();
        dynamic(move || {
            let glyph = row_glyph(store, &id)?;
            let group = store.is_group(&id);
            let said = store.source_tag(&id);
            Some(untracked(move || {
                let mut marks = view()
                    .flex_row()
                    .items_center()
                    .role(AccessibilityRole::Image)
                    .label(said)
                    .gap_1()
                    .child(ui::icon_inline(glyph).text_color(colors::FAINT));
                if group {
                    marks = marks.child(ui::icon_inline(Glyph::Group).text_color(colors::FAINT));
                }
                marks
            }))
        })
    };
    if !ui::touch() {
        return view()
            .flex_1()
            .min_w(px(0.0))
            .flex_row()
            .items_center()
            .gap_2()
            .child(title)
            .child(star)
            .child(tag_text().max_w(percentage(0.4)))
            .child(mark());
    }
    let said = move || !row_tag(store, &any_id).is_empty() || row_glyph(store, &any_id).is_some();
    view()
        .flex_1()
        .min_w(px(0.0))
        .gap(px(2.0))
        .child(
            view()
                .flex_row()
                .items_center()
                .gap_2()
                .child(title)
                .child(star),
        )
        .child(when(said, move || {
            view()
                .flex_row()
                .items_center()
                .gap_1()
                .child(tag_text().min_w(px(0.0)))
                .child(mark())
        }))
}

/// The words a conversation row is tagged with: its project's name outside
/// that project's page, else the chat it came from when it has no icon.
fn row_tag(store: Store, id: &str) -> String {
    let tag = row_place(store, id);
    // Found by a search among the others: it says it was put away.
    if store.is_archived(id) && !store.showing_archived() {
        return match tag.as_str() {
            "" => "Archived".to_owned(),
            tag => format!("Archived · {tag}"),
        };
    }
    tag
}

fn row_place(store: Store, id: &str) -> String {
    if let Some(project) = store.session_project(id).and_then(|p| store.project(&p)) {
        return match store.sidebar_view() {
            SidebarView::Project(_) => String::new(),
            _ => project.name,
        };
    }
    match source_glyph(store.source(id)) {
        Some(_) => String::new(),
        None => store.source_tag(id),
    }
}

/// The chat icon a conversation row shows: none once it is in a project.
fn row_glyph(store: Store, id: &str) -> Option<Glyph> {
    let in_project = store
        .session_project(id)
        .and_then(|p| store.project(&p))
        .is_some();
    source_glyph(store.source(id)).filter(|_| !in_project)
}

/// The icon of a chat platform a conversation came from; none for the app
/// or a platform without one.
fn source_glyph(source: Source) -> Option<Glyph> {
    match source {
        Source::Telegram => Some(Glyph::Telegram),
        Source::Discord => Some(Glyph::Discord),
        Source::Slack => Some(Glyph::Slack),
        _ => None,
    }
}

/// A source filter's icon: every inbox for All, the app's chat, the
/// platform's mark, people for groups.
fn filter_glyph(source: Source) -> Option<Glyph> {
    match source {
        Source::All => Some(Glyph::Inbox),
        Source::App | Source::Chat => Some(Glyph::Chat),
        Source::Groups => Some(Glyph::Group),
        source => source_glyph(source),
    }
}

/// What a conversation's row says at its end.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RowState {
    NeedsYou,
    Working,
    WasWorking,
    Quiet,
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
                        "AIDaemon"
                    };
                    let title = store.title(&hit.session_id);
                    let title = if store.is_archived(&hit.session_id) {
                        format!("{title} (archived)")
                    } else {
                        title
                    };
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

/// The connection's dot colour and words.
fn felt_status(felt: &Felt) -> (Tone, String) {
    match felt {
        Felt::Online => (Tone::Success, "Connected to the daemon".into()),
        Felt::Connecting => (Tone::Warning, "Connecting…".into()),
        Felt::Reconnecting => (Tone::Warning, "Reconnecting…".into()),
        Felt::NeedsPairing => (Tone::Warning, "Not paired yet".into()),
        Felt::NoDaemon => (Tone::Warning, "No daemon yet".into()),
        Felt::Down { kind, reason } => (
            Tone::Danger,
            match kind {
                DropKind::DaemonNotRunning => "Can't reach the daemon".into(),
                DropKind::Unauthorized => format!("The daemon refused this app: {reason}"),
                DropKind::VersionUnsupported => "The daemon needs a newer app".into(),
                DropKind::Dropped => "Connection lost".into(),
            },
        ),
    }
}

fn connection(store: Store) -> View {
    view().px_4().py_2().child(dynamic(move || {
        let felt = store.felt();
        Some(untracked(move || match felt {
            Felt::Online => status("Connected to the daemon", Tone::Success),
            Felt::Connecting => status("Connecting…", Tone::Accent),
            Felt::NeedsPairing => status("Not paired yet", Tone::Warning),
            Felt::NoDaemon => status("No daemon yet", Tone::Warning),
            // A few seconds in: not alarm yet, the next try is near.
            Felt::Reconnecting => status("Reconnecting…", Tone::Accent),
            Felt::Down { kind, reason } => {
                let headline = match kind {
                    DropKind::DaemonNotRunning => "Can't reach the daemon",
                    DropKind::Unauthorized => "The daemon refused this app",
                    DropKind::VersionUnsupported => "The daemon needs a newer app",
                    DropKind::Dropped => "Connection lost",
                };
                // One row: the footer sits under the conversation list and
                // must not eat its room on a phone.
                let detail = if kind == DropKind::Unauthorized {
                    reason
                } else {
                    "Reconnecting…".to_string()
                };
                // A paired key the daemon refused (revoked, or the daemon
                // was reset) never comes back on its own: offer to pair
                // again.
                let repair = kind == DropKind::Unauthorized && crate::link::signs_in_paired();
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
                    .child(when(repair, move || {
                        ui::btn(
                            "Pair again",
                            ui::Variant::Secondary,
                            ui::Size::Sm,
                            move || crate::link::pair_again(store),
                        )
                    }))
            }
        }))
    }))
}

/// The archived conversations, as a page of the sidebar: Back, how one is
/// brought back, then the conversations. `project`: only that project's,
/// asked for on its page.
fn archived_page(store: Store, project: Option<String>, ids: Memo<Vec<String>>) -> View {
    let named = project.clone();
    let ids = computed(move || {
        ids.with(|ids| {
            ids.iter()
                .filter(|s| project.is_none() || store.session_project(s) == project)
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    view()
        .gap(px(1.0))
        .child(page_head(
            store,
            move || match named.as_ref().and_then(|id| store.project(id)) {
                Some(project) => format!("Archived in {}", project.name),
                None => "Archived".to_owned(),
            },
            view(),
        ))
        .child(
            text(move || {
                if ids.with(Vec::is_empty) {
                    "Nothing is archived."
                } else if ui::touch() {
                    "Hold a conversation to bring it back."
                } else {
                    "Right-click a conversation to bring it back."
                }
            })
            .px_2()
            .py_1()
            .text_size(px(type_scale::SM))
            .text_color(colors::FAINT),
        )
        .child(listed(store, ids))
}
