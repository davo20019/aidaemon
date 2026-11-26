//! The sidebar. Home: its name, Search (a field once pressed) and one New
//! conversation, the conversations
//! that need the owner or are working, the pinned ones, the projects used
//! last (one line each; a press goes inside one), then every conversation
//! newest first, each with its time and tagged with its project or the
//! chat it came from. A project's
//! page: its folder, its conversations, New conversation there, Edit.
//! Then the pages and, when something is wrong, the connection.

use nsh_core::prelude::*;

use crate::store::{AnswerSurface, Felt, Page, SearchRead, SidebarView, Source, Store, now_ms};
use crate::ui::{self, Glyph, Tone, colors, icon, sidebar_row, sidebar_section_header, status};
use aidaemon_client_core::machine::DropKind;
use aidaemon_client_protocol::{Approval, ApprovalDecision};

pub fn sidebar(store: Store) -> View {
    // While the pointer is over the list its rows hold still: a row that
    // moved under the pointer as another conversation got busy would take
    // the click meant for its neighbour. Leaving the list shows the order
    // as it is.
    let held = signal(false);
    let shown = std::rc::Rc::new(std::cell::RefCell::new(Vec::<String>::new()));
    let ids = computed(move || {
        let fresh = store.visible_ids();
        let next = if held.get() {
            hold_order(&shown.borrow(), fresh)
        } else {
            fresh
        };
        shown.borrow_mut().clone_from(&next);
        next
    });
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
        // Compact drawers draw their exposed edge via `ui::drawer` (`BORDER`);
        // on desktop the docked sidebar keeps a quiet `LINE` separator.
        .style_when(
            move || !store.is_compact(),
            |s| s.border_r_1().border_color(colors::LINE),
        )
        // The top line: the name, connection dot (opens System), Search,
        // (a phone's) New conversation and Close; Search turns the line into
        // the field, Cancel turns it back.
        .child(top_line(store, query))
        .child(when(!ui::touch(), move || {
            view().px(px(ui::fit::sidebar_list_inset())).pt_1().child(
                sidebar_row(
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
                    .text_size(px(ui::fit::body())),
                )
                .child(ui::kbd(key_label("mod+n"))),
            )
        }))
        // Pinned subview header (All projects, a project, or Archived) so deep
        // scrolling never loses the Back button or page title.
        .child(subview_head(store))
        .child(ui::refreshable(
            scroll().flex_1().child(
                view()
                    .px(px(ui::fit::sidebar_list_inset()))
                    .pb_4()
                    .gap(px(1.0))
                    .on_mouse_enter(move |_: &MouseEvent| held.set(true))
                    .on_mouse_leave(move |_: &MouseEvent| held.set(false))
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
                    // Only show a local empty-title message when message search
                    // has not run; once message search is Ready, `search_hits`
                    // owns the combined empty state so it never contradicts hits.
                    .child(when(
                        move || {
                            ids.with(Vec::is_empty)
                                && !store.filter().trim().is_empty()
                                && matches!(store.search(), SearchRead::Idle)
                        },
                        || {
                            text("No conversation matches.")
                                .px(px(ui::fit::sidebar_row_inset()))
                                .py_2()
                                .text_size(px(ui::fit::small()))
                                .text_color(colors::FAINT)
                        },
                    )),
            ),
            move || store.pull_refreshing(),
            move || store.pull_refresh(),
        ))
        // The connection line shows only when something is wrong: the dot by
        // the name opens System when healthy.
        .child(
            view()
                .style_when(
                    move || {
                        matches!(
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

/// The list in the order it was shown, `fresh` being the order now: rows
/// still listed keep their places, rows gone drop out and new ones go at
/// the end, out of the way until the list is let go.
fn hold_order(shown: &[String], fresh: Vec<String>) -> Vec<String> {
    let now: std::collections::HashSet<&String> = fresh.iter().collect();
    let mut held: Vec<String> = shown
        .iter()
        .filter(|id| now.contains(id))
        .cloned()
        .collect();
    let kept: std::collections::HashSet<String> = held.iter().cloned().collect();
    held.extend(fresh.into_iter().filter(|id| !kept.contains(id)));
    held
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
        .otherwise(move || {
            let work = shown.clone();
            view()
                .child(menu_row(store, shown.clone(), dated))
                .child(work_line(store, work))
        }),
    )
}

/// Under a conversation's row, while it has agents at work: what waits on
/// the owner (an approval, answered here), else a failure, else how many
/// agents of each kind and how far along ("1 of 3 done · Claude Code ×2,
/// Codex"). Steps stay in the conversation; a tap opens its agents.
fn work_line(store: Store, id: String) -> View {
    let ask_id = id.clone();
    let asking = computed(move || store.approvals_for(&ask_id).into_iter().next());
    let open_id = id.clone();
    let work_id = id;
    let work = computed(move || store.agent_work(&work_id));
    view().child(dynamic(move || {
        if let Some(approval) = asking.get() {
            return Some(untracked(move || ask_line(store, approval)));
        }
        let work = work.get()?;
        let open_id = open_id.clone();
        Some(untracked(move || {
            let (words, color) = work_words(&work);
            let began = work.began_at_ms;
            let label = format!("{words}, open its agents");
            let btn = nsh_base::button()
                .w_full()
                .accessibility_label(label)
                .rounded(px(ui::radius::CONTROL))
                .active(|s| s.bg(colors::SELECTED))
                .focus_visible(|s| s.outline(px(2.0)).outline_color(colors::ACCENT))
                .on_press(move || store.open_agents(open_id.clone()));
            let btn = if ui::touch() {
                btn
            } else {
                btn.hover(|s| s.bg(colors::HOVER))
            };
            view().child(
                btn.child(
                    view()
                        .w_full()
                        .min_h(px(ui::Size::Sm.height()))
                        .flex_row()
                        .items_center()
                        .gap_2()
                        .pl(px(ui::fit::sidebar_body_indent()))
                        .pr(px(ui::fit::sidebar_row_inset()))
                        .py_1()
                        .child(work_marks(store, &work, words, color))
                        .child(
                            text(move || {
                                store.tick();
                                ui::time::span(now_ms().saturating_sub(began) / 1000)
                            })
                            .tabular_nums()
                            .text_size(px(ui::fit::meta()))
                            .text_color(colors::FAINT),
                        ),
                ),
            )
        }))
    }))
}

/// The work line's body: a failure in words; else how far along, then
/// each kind of agent by its mark with how many ("1 of 3 done · ◆×2 ●"),
/// a lone agent by its mark and name. The words stay the line's label.
fn work_marks(
    store: Store,
    work: &crate::store::AgentWork,
    words: String,
    color: AdaptiveColor,
) -> View {
    let size = px(ui::fit::meta());
    let line = view()
        .flex_row()
        .items_center()
        .gap_1()
        .flex_1()
        .min_w(px(0.0));
    if !work.failed.is_empty() {
        return line.child(
            text(words)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(1)
                .text_size(size)
                .text_color(color),
        );
    }
    let mut line = line;
    if work.total > 1 {
        line = line.child(
            text(format!("{} of {} done ·", work.done, work.total))
                .flex_shrink(0.0)
                .text_size(size)
                .text_color(color),
        );
    }
    let alone = work.kinds.len() == 1 && work.total == 1;
    for (name, count) in &work.kinds {
        let glyph = store.agent_glyph_named(name);
        line = line.child(ui::icon_inline(glyph).text_color(color));
        if alone {
            line = line.child(
                text(name.clone())
                    .min_w(px(0.0))
                    .number_of_lines(1)
                    .text_size(size)
                    .text_color(color),
            );
        } else if *count > 1 {
            line = line.child(text(format!("×{count}")).text_size(size).text_color(color));
        }
    }
    line
}

/// The work line's words and colour: a failure first, else progress.
fn work_words(work: &crate::store::AgentWork) -> (String, AdaptiveColor) {
    if let Some(failed) = work.failed.first() {
        let running = work.total - work.done;
        let words = match running {
            0 => format!("{failed} failed"),
            n => format!("{failed} failed · {n} still running"),
        };
        return (words, colors::DANGER);
    }
    let kinds = work
        .kinds
        .iter()
        .map(|(name, count)| match count {
            1 => name.clone(),
            n => format!("{name} ×{n}"),
        })
        .collect::<Vec<_>>()
        .join(", ");
    // Progress first: a narrow row cuts the end, never the count.
    let words = if work.total > 1 {
        format!("{} of {} done · {kinds}", work.done, work.total)
    } else {
        kinds
    };
    (words, colors::MUTED)
}

/// A waiting approval under its conversation's row: who asks and what,
/// and Allow once / Deny when the card offers them, answered from here.
fn ask_line(store: Store, approval: Approval) -> View {
    let words = match approval
        .card
        .as_ref()
        .and_then(|card| card.asked_by.clone())
    {
        Some(who) => format!("{who}: {}", approval.summary),
        None => approval.summary.clone(),
    };
    let mut line = view()
        .flex_row()
        .items_center()
        .gap_2()
        .pl(px(ui::fit::sidebar_body_indent()))
        .pr(px(ui::fit::sidebar_row_inset()))
        .pb_1()
        .child(
            text(words)
                .flex_1()
                .min_w(px(0.0))
                .number_of_lines(2)
                .text_size(px(ui::fit::meta()))
                .text_color(colors::WARNING),
        );
    store.answer_shown(AnswerSurface::Line, &approval.id);
    for (decision, label, variant) in [
        (ApprovalDecision::ApproveOnce, "Allow", ui::Variant::Primary),
        (ApprovalDecision::Deny, "Deny", ui::Variant::Ghost),
    ] {
        if approval.options.contains(&decision) {
            let id = approval.id.clone();
            line = line.child(ui::btn(label, variant, ui::Size::Sm, move || {
                store.resolve_clicked(AnswerSurface::Line, &id, decision);
            }));
        }
    }
    line
}

/// The row with its menu.
fn menu_row(store: Store, id: String, dated: bool) -> View {
    let (open, pin, stop) = (id.clone(), id.clone(), id.clone());
    let pinned = untracked(|| store.is_pinned(&id));
    let current = untracked(|| store.session_project(&id));
    let mut actions = vec![
        ui::MenuAction::new("Open", move || {
            let open = open.clone();
            store.leave_drawer(move || store.select(Some(open)))
        }),
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
    {
        let session = id.clone();
        actions.push(
            ui::MenuAction::new("Delete…", move || confirm_delete_session(store, &session))
                .shown_when(move || store.can_delete_sessions()),
        );
    }
    let (preview, title, open) = (id.clone(), id.clone(), id.clone());
    let is_open = computed(move || store.selected().as_deref() == Some(open.as_str()));
    // While the menu is up its row is the marked one, whichever
    // conversation is open behind it.
    let about = signal(false);
    ui::with_menu_about(
        move || store.title(&title),
        about,
        ui::with_preview(
            session_button(store, id, dated).style_when(move || about.get(), ui::menu_target),
            move || session_preview(store, preview),
            // The open conversation is on the page already.
            move || is_open.get(),
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
/// who answers it, when it last moved, and its last message.
fn session_preview(store: Store, id: String) -> View {
    let (title, place, agent, when, last) = (id.clone(), id.clone(), id.clone(), id.clone(), id);
    view()
        .gap_1()
        .child(
            text(move || store.title(&title))
                .number_of_lines(2)
                .text_size(px(ui::fit::small()))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(colors::TEXT),
        )
        .child(
            text(move || {
                let mut parts = vec![store.place(&place)];
                if let Some(agent) = store.session_agent(&agent) {
                    parts.push(agent);
                }
                if let Some(at) = store.last_activity(&when) {
                    parts.push(ui::time::ago(at, crate::store::now_ms()));
                }
                parts.retain(|p| !p.is_empty());
                parts.join(" · ")
            })
            .number_of_lines(1)
            .text_size(px(ui::fit::meta()))
            .text_color(colors::FAINT),
        )
        .child(
            text(move || match store.last_message(&last) {
                Some((true, text)) => format!("You: {}", text.trim()),
                Some((false, text)) => text.trim().to_owned(),
                None => "Open it to read it.".to_owned(),
            })
            .number_of_lines(4)
            .text_size(px(ui::fit::meta()))
            .text_color(colors::MUTED),
        )
}

/// The sidebar's top line. Shut: "AIDaemon", Search, New conversation
/// (a phone's; a window has its own row) and Close (or ⌘K's hint). Open:
/// the search field across the line, focused, and Cancel, which clears it.
/// A search left in the field keeps it open.
/// The name button's own padding, taken from the line's so the name stays
/// aligned with the 16 pt sidebar content rail.
const NAME_INSET: f32 = 4.0;

fn top_line(store: Store, query: Signal<String>) -> View {
    let open = signal(!untracked(|| query.get()).trim().is_empty());
    let close = move || {
        query.set(String::new());
        open.set(false);
    };
    let header_action_size = if ui::touch() {
        ui::Size::Sm
    } else {
        ui::Size::Md
    };
    view()
        .min_h(px(ui::fit::bar()))
        .flex_row()
        .items_center()
        .pl(px(16.0 - NAME_INSET))
        // Touch: 2 pt right edge + 44 pt Close button centres × at 24 pt,
        // matching ⋮ below and the 22 pt grip/chevron rail. Desktop: 16 pt
        // right edge aligns ⌘K cleanly with timestamps.
        .pr(px(if ui::touch() {
            2.0
        } else {
            ui::fit::sidebar_list_inset() + ui::fit::sidebar_row_inset()
        }))
        .gap_1()
        .border_b_1()
        .border_color(colors::LINE)
        .child(
            when(
                move || open.get(),
                move || {
                    view()
                        .flex_1()
                        .flex_row()
                        .items_center()
                        .gap_1()
                        .pr(px(if ui::touch() { 6.0 } else { 0.0 }))
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
                    .gap(px(if ui::touch() { 0.0 } else { 4.0 }))
                    // The name and connection dot sit tightly together as one
                    // unit (~8 pt gap): the name goes home; the dot opens System.
                    .child(
                        view()
                            .flex_row()
                            .items_center()
                            .child(ui::name_btn("AIDaemon", NAME_INSET, move || {
                                store.show_sidebar_view(SidebarView::Home);
                                store.leave_drawer(move || store.select(None));
                            }))
                            // Only when something is wrong: connected needs no
                            // mark, and the dot then opens System.
                            .child(when(
                                move || !matches!(store.felt(), Felt::Online),
                                move || {
                                    ui::status_dot(
                                        move || felt_status(&store.felt()).0,
                                        move || {
                                            format!("{}, open System", felt_status(&store.felt()).1)
                                        },
                                        move || store.leave_drawer(move || store.open_system()),
                                    )
                                },
                            )),
                    )
                    .child(view().flex_1())
                    .child(settings_button(store, header_action_size))
                    .child(ui::icon_btn(
                        Glyph::Search,
                        "Search conversations",
                        header_action_size,
                        move || open.set(true),
                    ))
                    .child(when(ui::touch(), move || {
                        ui::icon_btn(
                            Glyph::Compose,
                            "New conversation",
                            header_action_size,
                            move || new_here(store),
                        )
                    }))
                    // In a drawer it closes with a swipe toward its edge, a
                    // press beside it or the back button: no close button.
                    .child(when(
                        move || !store.is_compact(),
                        || ui::kbd(key_label("mod+k")),
                    ))
            }),
        )
}

/// New conversation from the sidebar: inside a project's page it starts
/// there. At once, not after a drawer's slide (`Store::leave_drawer`):
/// its composer must take the focus inside the tap, the only place iOS
/// Safari raises the keyboard for it; the page is a light one.
fn new_here(store: Store) {
    match untracked(|| store.sidebar_view()) {
        SidebarView::Project(id) => store.new_in_project(Some(id)),
        // From a conversation in a project: a new one on code in the same
        // project's folder.
        _ => match untracked(|| store.current_project()) {
            Some(project) => store.new_in_project(Some(project)),
            None => store.with_keyboard(|| store.select(None)),
        },
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
    let total_projects = computed(move || all.with(Vec::len));
    let reorder = Reorder::new(shown, false);
    view()
        .gap(px(1.0))
        .child(group(store, "Needs you", needs))
        .child(group(store, "Pinned", pinned))
        .child(sidebar_section_header("Projects", None))
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
                sidebar_row(
                    || false,
                    move || store.show_sidebar_view(SidebarView::Projects),
                )
                .min_h(px(ui::fit::compact_row()))
                .accessibility_label("All projects")
                .child(icon(Glyph::Grid).text_color(colors::FAINT))
                .child(
                    text("All projects")
                        .flex_1()
                        .text_size(px(ui::fit::body()))
                        .text_color(colors::MUTED),
                )
                .child(
                    text(move || total_projects.get().to_string())
                        .tabular_nums()
                        .text_size(px(ui::fit::meta()))
                        .text_color(colors::FAINT),
                )
                .child(ui::icon_inline(Glyph::Chevron).text_color(colors::FAINT))
            })
            .otherwise(move || new_project_row(store)),
        )
        // Their heading, with which ones are listed as one filter: where
        // they came from (when from more than one place) and the archived
        // (when there are any).
        .child(sidebar_section_header(
            "Recent",
            Some(view().child(when(
                move || !store.sources().is_empty() || store.archived_count() > 0,
                move || source_filter(store),
            ))),
        ))
        .child(listed(store, recent))
        .child(when(
            move || recent.with(Vec::is_empty) && store.source_filter() != Source::All,
            || {
                text("None from here.")
                    .px(px(ui::fit::sidebar_row_inset()))
                    .py_1()
                    .text_size(px(ui::fit::small()))
                    .text_color(colors::FAINT)
            },
        ))
}

/// A drag reordering a list of projects (Home's, or every project's):
/// which one and how far it moved, the list as it stands, and how tall its
/// rows are (`detailed`: a second line, the folder). `measured_row` tracks
/// the actual laid-out row height so drag landing stays accurate under large
/// system text sizes.
#[derive(Clone, Copy)]
struct Reorder {
    drag: Signal<Option<(String, f32)>>,
    order: Memo<Vec<String>>,
    detailed: bool,
    measured_row: Signal<f32>,
}

/// The gap between two project rows.
const ROW_GAP: f32 = 1.0;

impl Reorder {
    fn new(order: Memo<Vec<String>>, detailed: bool) -> Self {
        let initial = ui::fit::compact_row()
            + if detailed {
                if ui::touch() { 18.0 } else { 14.0 }
            } else {
                0.0
            };
        Self {
            drag: signal(None),
            order,
            detailed,
            measured_row: signal(initial),
        }
    }

    /// A row's minimum height, falling back to the measured height when OS
    /// font scaling grows the row past its baseline.
    fn min_row(&self) -> f32 {
        ui::fit::compact_row()
            + if self.detailed {
                if ui::touch() { 18.0 } else { 14.0 }
            } else {
                0.0
            }
    }

    /// Current laid-out row height for drag stride and handle height.
    fn row(&self) -> f32 {
        self.measured_row.get().max(self.min_row())
    }

    fn note_row_height(&self, height: f32) {
        if height > 0.0 && (self.measured_row.get_untracked() - height).abs() > 0.5 {
            self.measured_row.set(height);
        }
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
/// quiet inline drag handle (and `detailed` shows its folder).
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
        // At once, as `new_here`: the keyboard comes up inside the tap.
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
    let mut names = view().flex_1().min_w(px(0.0)).gap(px(2.0)).child(
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
            .text_size(px(ui::fit::meta()))
            .text_color(colors::FAINT),
        );
    }
    let star = pinned.clone();
    let mut line = sidebar_row(
        || false,
        move || store.show_sidebar_view(SidebarView::Project(open.clone())),
    )
    .min_h(px(ui::fit::compact_row()))
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
    .child(when(star, || {
        ui::icon_inline(Glyph::Pinned).text_color(colors::FAINT)
    }))
    .child(
        view().flex_row().justify_end().min_w(px(20.0)).child(
            text(move || match count.get() {
                0 => String::new(),
                n => n.to_string(),
            })
            .tabular_nums()
            .text_size(px(ui::fit::meta()))
            .text_color(colors::FAINT),
        ),
    );
    let about = signal(false);
    line = line.style_when(move || about.get(), ui::menu_target);
    let about_id = id.clone();
    let about_name = move || store.project(&about_id).map(|p| p.name).unwrap_or_default();
    let Some(reorder) = reorder else {
        return ui::with_menu_about(
            about_name,
            about,
            line.child(ui::icon_inline(Glyph::Chevron).text_color(colors::FAINT)),
            actions,
        );
    };
    // Tighten inner right padding before the sibling drag handle so the count
    // sits closer to the grip and the grip centres on the chevron rail.
    line = line.pr_0().min_h(px(reorder.min_row()));
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
        .on_layout(move |bounds: &Bounds<Points>| reorder.note_row_height(bounds.size.height.0))
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
        // 36 pt visual lane + 4 pt hit_slop on each side = 44 pt touch target,
        // centring the inline grip on the same right rail as All projects' chevron.
        .w(px(if ui::touch() { 36.0 } else { 24.0 }))
        .h(move || px(reorder.row()))
        .hit_slop(EdgeInsets::all(px(if ui::touch() { 4.0 } else { 0.0 })))
        .items_center()
        .justify_center()
        // The mouse's and a finger's; the keyboard and a screen reader
        // have Move up and Move down in the row's menu. Kept at full FAINT
        // token contrast (no extra opacity dimming) so contrast stays >= 4.5:1.
        .child(ui::icon_inline(Glyph::Grip).text_color(colors::FAINT))
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

/// Ask, then delete a conversation for good, as `/wipe` erases one.
fn confirm_delete_session(store: Store, id: &str) {
    let title = untracked(|| store.title(id));
    let id = id.to_owned();
    ui::confirm(
        format!("Delete {title}?"),
        "Its messages, runs and summaries are erased for good, and its running work stops. \
         Facts already saved to memory are kept."
            .to_owned(),
        "Delete",
        move || store.delete_session(&id),
    );
}

/// "New project": opens the form.
fn new_project_row(store: Store) -> View {
    view().child(
        sidebar_row(|| false, move || store.show_project_form(true))
            .min_h(px(ui::fit::compact_row()))
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

/// Pinned subview header rendered above the scroll area for All projects,
/// a project page, or Archived, so deep scrolling keeps Back and the title
/// in view.
fn subview_head(store: Store) -> View {
    view().child(dynamic(move || {
        if !store.filter().trim().is_empty() {
            return None;
        }
        let archived = store.showing_archived();
        let shown = store.sidebar_view();
        if archived {
            let project = match shown {
                SidebarView::Project(id) => Some(id),
                _ => None,
            };
            return Some(untracked(move || {
                page_head(
                    store,
                    move || match project.as_ref().and_then(|id| store.project(id)) {
                        Some(project) => format!("Archived in {}", project.name),
                        None => "Archived".to_owned(),
                    },
                    view(),
                )
            }));
        }
        match shown {
            SidebarView::Home => None,
            SidebarView::Projects => Some(untracked(move || {
                page_head(store, || "Projects".to_owned(), view())
            })),
            SidebarView::Project(id) => Some(untracked(move || {
                let (name, edit, put_away) = (id.clone(), id.clone(), id.clone());
                let (pin, unpin, pinned, unpinned, gone) =
                    (id.clone(), id.clone(), id.clone(), id.clone(), id.clone());
                let archived_btn = dynamic(move || {
                    let any = store.session_ids().iter().any(|s| {
                        store.is_archived(s)
                            && store.session_project(s).as_deref() == Some(put_away.as_str())
                    });
                    any.then(|| {
                        untracked(|| {
                            ui::icon_btn(
                                Glyph::Archive,
                                "Archived conversations in this project",
                                ui::Size::Sm,
                                move || store.show_archived(true),
                            )
                        })
                    })
                });
                let tools =
                    view()
                        .flex_row()
                        .items_center()
                        .child(archived_btn)
                        .child(ui::press_menu(
                            Glyph::More,
                            "Project options",
                            vec![
                                ui::MenuAction::new("Pin to the top", move || {
                                    store.toggle_project_pin(&pin)
                                })
                                .shown_when(move || !store.is_project_pinned(&unpinned)),
                                ui::MenuAction::new("Unpin", move || {
                                    store.toggle_project_pin(&unpin)
                                })
                                .shown_when(move || store.is_project_pinned(&pinned)),
                                ui::MenuAction::new("Edit name or folder", move || {
                                    store.edit_project(&edit)
                                }),
                                ui::MenuAction::new("Delete project…", move || {
                                    confirm_delete(store, &gone)
                                }),
                            ],
                        ));
                page_head(
                    store,
                    move || store.project(&name).map(|p| p.name).unwrap_or_default(),
                    tools,
                )
            })),
        }
    }))
}

/// The top of a page inside the sidebar: Back, its title and what it
/// offers. Back is 44 pt on touch starting at 4 pt so the title lands on the
/// 48 pt inner rail beside the folder/New-conversation labels below it.
fn page_head(store: Store, title: impl Fn() -> String + 'static, tools: View) -> View {
    let in_project = untracked(|| {
        store.showing_archived() && matches!(store.sidebar_view(), SidebarView::Project(_))
    });
    view()
        .flex_row()
        .items_center()
        .gap(px(if ui::touch() { 0.0 } else { 4.0 }))
        .pl(px(if ui::touch() {
            4.0
        } else {
            ui::fit::sidebar_list_inset()
        }))
        .pr(px(if ui::touch() {
            2.0
        } else {
            ui::fit::sidebar_list_inset()
        }))
        .py(px(2.0))
        .border_b_1()
        .border_color(colors::LINE)
        .child(ui::icon_btn(
            Glyph::Back,
            if in_project {
                "Back to the project"
            } else {
                "Back to all conversations"
            },
            ui::Size::Sm,
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
        .pt_1()
        .gap(px(1.0))
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

/// One project: its folder, Edit, then its conversations
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
    let (folder, empty) = (id.clone(), id.clone());
    let (browse, browsing, has_folder) = (id.clone(), id.clone(), id);
    view()
        .pt_1()
        .gap(px(1.0))
        // Its folder, the way into its files (a page of their own, no
        // conversation needed); without one, the way to add one.
        .child(dynamic(move || {
            let path = store
                .project(&folder)
                .and_then(|p| p.path)
                .filter(|f| !f.trim().is_empty());
            let (open, on, add) = (browse.clone(), browsing.clone(), has_folder.clone());
            Some(untracked(move || match path {
                Some(path) => sidebar_row(
                    move || store.showing_files_of(&on),
                    move || store.open_project_files(&open),
                )
                .min_h(px(ui::fit::compact_row()))
                .accessibility_label("This project's files")
                .child(icon(Glyph::Folder).text_color(colors::MUTED))
                .child(
                    text(short_path(&path))
                        .flex_1()
                        .min_w(px(0.0))
                        .number_of_lines(1)
                        .font_family(ui::theme::MONO)
                        .text_size(px(ui::fit::small()))
                        .text_color(colors::MUTED),
                )
                .child(ui::icon_inline(Glyph::Chevron).text_color(colors::FAINT)),
                None => sidebar_row(|| false, move || store.edit_project(&add))
                    .min_h(px(ui::fit::compact_row()))
                    .accessibility_label("Add a folder to this project")
                    .child(icon(Glyph::Folder).text_color(colors::FAINT))
                    .child(
                        text("Add a folder…")
                            .flex_1()
                            .number_of_lines(1)
                            .text_size(px(ui::fit::small()))
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
            // At once, as `new_here`: the keyboard comes up inside the tap.
            sidebar_row(|| false, move || store.new_in_project(Some(start.clone())))
                .min_h(px(ui::fit::compact_row()))
                .accessibility_label("New conversation in this project")
                .child(icon(Glyph::Compose).text_color(colors::MUTED))
                .child(
                    text("New conversation")
                        .flex_1()
                        .number_of_lines(1)
                        .text_size(px(ui::fit::body())),
                )
        }))
        .child(when(
            move || mine.with(Vec::is_empty),
            move || {
                text("No conversations yet")
                    .px(px(ui::fit::sidebar_row_inset()))
                    .py_1()
                    .text_size(px(ui::fit::small()))
                    .text_color(colors::FAINT)
            },
        ))
        .child(when(
            move || !mine.with(Vec::is_empty),
            || sidebar_section_header("Conversations", None),
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
            .min_h(px(if ui::touch() { 32.0 } else { 24.0 }))
            .hit_slop(EdgeInsets::all(px(if ui::touch() { 6.0 } else { 0.0 })))
            .flex_row()
            .items_center()
            .gap_1()
            .px_2()
            .child(dynamic(move || {
                let glyph = filter_glyph(store.source_filter())?;
                Some(untracked(move || {
                    ui::icon_inline(glyph)
                        .text_color(colors::MUTED)
                        .accessibility_elements_hidden(true)
                }))
            }))
            .child(
                text(shown)
                    .text_size(px(ui::fit::meta()))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(colors::MUTED),
            )
            .child(
                ui::icon_inline(Glyph::Down)
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
                .text_size(px(ui::fit::meta()))
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
    let row = sidebar_row(
        // Lit only while the conversation is what the page shows, not
        // behind another page (Files, Memory, …).
        move || {
            store.page() == Page::Conversation && store.selected().as_deref() == Some(a.as_str())
        },
        move || {
            let b = b.clone();
            store.leave_drawer(move || store.select(Some(b)))
        },
    )
    // One-line conversations sit at 44 pt on touch (matching Projects);
    // two-line rows grow naturally to ~54–56 pt from their content.
    .min_h(px(ui::fit::compact_row()))
    .py(px(if ui::touch() { 6.0 } else { 5.0 }))
    // Read as one item: its name, where it lives, who answers and whether it is working.
    .accessibility_label(move || {
        let id = &label_id;
        let mut parts = vec![store.title(id), store.place(id)];
        if let Some(agent) = store.session_agent(id) {
            parts.push(agent);
        }
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
    });
    // Desktop keeps the 16 pt leading dot slot so project names and
    // conversation titles stay aligned at 40 pt, top-aligned to line 1 so the
    // dot shares line 1's baseline with the title and timestamp. On touch, the
    // dot tucks into the row's 12 pt left padding on line 1.
    let row = if !ui::touch() {
        row.items_start().child(
            ui::state_dot(
                move || store.is_unread(&c),
                move || {
                    if store.is_unread(&e) {
                        colors::TEXT
                    } else {
                        colors::TRANSPARENT
                    }
                },
            )
            .h(px(19.0)),
        )
    } else {
        row
    };
    row.child(row_names(store, id, d, f, g, h, date_id, dated))
}

/// A conversation row's name, trailing state/timestamp on line 1, and where it
/// is from on line 2: its project's name (outside that project's page), its
/// coding agent when set to one, or the chat it came from when not the app
/// (Telegram, Discord and Slack by their icons, a group adding people; any
/// other by name). A row with nothing to say stays one line.
#[allow(clippy::too_many_arguments)]
fn row_names(
    store: Store,
    id: String,
    d: String,
    f: String,
    g: String,
    h: String,
    date_id: String,
    dated: bool,
) -> View {
    let unread_id = h.clone();
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
    let star = when(
        move || store.is_pinned(&f),
        || ui::icon_inline(Glyph::Pinned).text_color(colors::FAINT),
    );
    // At line 1's end, what it is doing: waiting on the owner ("Needs you",
    // amber), working (a turning ring), else when it last moved ("3m", "Nov 24")
    // where the list shows times. Aligned to the title line so 1-line and
    // 2-line rows match and metadata below gets the row's full width.
    let trailing = view().flex_shrink(0.0).child(dynamic(move || {
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
                .text_size(px(ui::fit::meta()))
                .text_color(colors::FAINT)
            })),
        }))
    }));
    let (tag_id, mark_id, any_id) = (g, id.clone(), id);
    let tag_shown_id = tag_id.clone();
    let agent_id = tag_id.clone();
    let agent_shown_id = tag_id.clone();
    let tag = move || row_tag(store, &tag_id);
    let has_tag = move || !row_tag(store, &tag_shown_id).is_empty();
    let tag_text = move || {
        let tag = tag.clone();
        text(tag)
            .number_of_lines(1)
            .text_size(px(ui::fit::meta()))
            .text_color(colors::MUTED)
    };
    // The coding agent the conversation is set to (messages there go
    // straight to it), by its maker's mark; the row's label names it.
    let has_agent = move || store.session_agent(&agent_shown_id).is_some();
    let agent_text = move || {
        let id = agent_id.clone();
        dynamic(move || {
            let glyph = store.session_agent_glyph(&id);
            Some(untracked(move || {
                ui::icon_inline(glyph)
                    .flex_shrink(0.0)
                    .text_color(colors::FAINT)
            }))
        })
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
    // Two lines everywhere: the title + time/state, then where it is and who
    // answers (one desktop line had no room for both beside a title).
    let said = move || {
        !row_tag(store, &any_id).is_empty()
            || row_glyph(store, &any_id).is_some()
            || store.session_agent(&any_id).is_some()
    };
    let mut top = view().flex_row().items_center().gap_2();
    if ui::touch() {
        top = top.child(ui::sidebar_unread_dot_inset(move || {
            store.is_unread(&unread_id)
        }));
    }
    top = top.child(title).child(star).child(trailing);
    view()
        .flex_1()
        .min_w(px(0.0))
        .gap(px(if ui::touch() { 3.0 } else { 2.0 }))
        .child(top)
        .child(when(said, move || {
            let has_tag = has_tag.clone();
            let tag_text = tag_text.clone();
            let has_agent = has_agent.clone();
            let agent_text = agent_text.clone();
            // The line shrinks with the row, and the project's name gives
            // way (one line, ending in "…") before the agent and source icons:
            // empty slots stay unmounted so they add no leading flex gap.
            view()
                .flex_row()
                .items_center()
                .min_w(px(0.0))
                .gap_1p5()
                .child(when(has_tag, move || {
                    tag_text().min_w(px(0.0)).flex_shrink(1.0)
                }))
                .child(when(has_agent, agent_text))
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
        // Global search always names the project even when started inside one.
        let searching = !store.filter().trim().is_empty();
        return match store.sidebar_view() {
            SidebarView::Project(_) if !searching => String::new(),
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

/// Settings as one icon in the sidebar's header, beside search: opens the
/// configuration hub (Settings, Automations, Memory and System, switched
/// from their shared top bar). Highlighted while any of the four is open,
/// with an accent dot when automations have unseen news. No row of its own:
/// the drawer's height is the conversations'.
fn settings_button(store: Store, size: ui::Size) -> impl IntoElement {
    let active = move || {
        matches!(
            store.page(),
            Page::Settings | Page::Automations | Page::Memory | Page::System
        )
    };
    ui::icon_nav_btn(
        Glyph::Settings,
        "Settings: change how the daemon runs",
        size,
        active,
        move || store.automation_news() > 0,
        move || store.leave_drawer(move || store.open_settings()),
    )
}

/// A titled group of conversations, shown only when it has any.
fn group(store: Store, title: &'static str, ids: Memo<Vec<String>>) -> View {
    view()
        .gap(px(1.0))
        .child(when(
            move || !ids.with(Vec::is_empty),
            move || sidebar_section_header(title, None),
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
            SearchRead::Loading(_) => {
                view().child(sidebar_section_header("In messages: searching…", None))
            }
            SearchRead::Failed(error) => view().child(
                text(error)
                    .px(px(ui::fit::sidebar_row_inset()))
                    .text_size(px(ui::fit::small()))
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
                let mut list = view().gap(px(1.0)).child(sidebar_section_header(
                    if hits.is_empty() {
                        "Nothing matches, in titles or messages"
                    } else {
                        "In messages"
                    },
                    None,
                ));
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
                        sidebar_row(
                            || false,
                            move || {
                                let open = open.clone();
                                store.leave_drawer(move || store.select(Some(open)))
                            },
                        )
                        // Align search results with the conversation title rail
                        // on both touch (16 pt) and desktop (40 pt).
                        .pl(px(ui::fit::sidebar_body_indent()))
                        .py(px(if ui::touch() { 6.0 } else { 5.0 }))
                        .accessibility_label(format!("{title}: {}", hit.snippet))
                        .child(
                            view()
                                .flex_1()
                                .min_w(px(0.0))
                                .gap(px(2.0))
                                .child(
                                    view()
                                        .flex_row()
                                        .items_center()
                                        .gap_2()
                                        .child(
                                            text(title)
                                                .flex_1()
                                                .min_w(px(0.0))
                                                .number_of_lines(1)
                                                .text_size(px(ui::fit::body()))
                                                .font_weight(FontWeight::MEDIUM)
                                                .text_color(colors::TEXT),
                                        )
                                        .child(
                                            text(ui::time::ago(hit.at_ms, now_ms()))
                                                .flex_shrink(0.0)
                                                .tabular_nums()
                                                .text_size(px(ui::fit::meta()))
                                                .text_color(colors::FAINT),
                                        ),
                                )
                                .child(
                                    text(format!("{who}: {}", hit.snippet))
                                        .number_of_lines(2)
                                        .text_size(px(ui::fit::small()))
                                        .text_color(colors::MUTED),
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
                // was reset), or a daemon not reached for long (it may have
                // moved where the saved address no longer finds it), does
                // not come back on its own: offer to pair again.
                let repair = crate::link::signs_in_paired() && store.may_pair_again();
                view()
                    .flex_row()
                    .items_center()
                    .gap_2()
                    .child(status(headline, Tone::Danger).flex_1().min_w(px(0.0)))
                    .child(
                        text(detail)
                            .number_of_lines(1)
                            .text_size(px(ui::fit::meta()))
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
    let ids = computed(move || {
        ids.with(|ids| {
            ids.iter()
                .filter(|s| project.is_none() || store.session_project(s) == project)
                .cloned()
                .collect::<Vec<_>>()
        })
    });
    view()
        .pt_1()
        .gap(px(1.0))
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
            .px(px(ui::fit::sidebar_row_inset()))
            .py_1()
            .text_size(px(ui::fit::small()))
            .text_color(colors::FAINT),
        )
        .child(listed(store, ids))
}

#[cfg(test)]
mod hold_tests {
    use super::hold_order;

    fn v(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn held_rows_keep_their_places_and_new_ones_go_last() {
        let shown = v(&["a", "b", "c", "d"]);
        // "c" got busy and moved up, "b" was archived, "e" is new.
        let fresh = v(&["e", "c", "a", "d"]);
        assert_eq!(hold_order(&shown, fresh), v(&["a", "c", "d", "e"]));
    }

    #[test]
    fn nothing_shown_yet_takes_the_fresh_order() {
        assert_eq!(hold_order(&[], v(&["b", "a"])), v(&["b", "a"]));
    }
}
