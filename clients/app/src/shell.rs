//! The window: sidebar, conversation, details (Changes and Steps), and the
//! command palette over them. Narrow windows fold the details away first,
//! then the sidebar. The terminal is a coding tool first: the conversation
//! fills it, the list and the details open on a key, and run in a
//! project's folder it starts in that project.

use aidaemon_client_protocol::ApprovalDecision;
use nsh_base::{CommandEntry, CommandItem, command, dialog};
use nsh_core::platform::Idiom;
use nsh_core::prelude::*;

use crate::features::{conversation, inspector::inspector, sidebar::sidebar};
use crate::store::{Connection, Store};
use crate::ui::{self, colors, radius, size, type_scale};

pub fn shell(store: Store) -> View {
    let terminal = idiom() == Idiom::Terminal;
    let wide = computed(move || window().width.0 >= 1180.0);
    let roomy = computed(move || window().width.0 >= 760.0);
    store.set_inspector(!terminal);
    if terminal {
        store.set_sidebar(false);
    }
    effect(move || store.set_compact(!roomy.get()));
    crate::features::notify::approvals(store);
    crate::features::notify::replies(store);
    set_interval(std::time::Duration::from_secs(1), move || {
        store.advance_tick()
    });
    let palette = signal(false);
    // Only entering and leaving setup remakes the page, not each step's news.
    let in_setup = computed(move || {
        store
            .setup()
            .is_some_and(|s| !matches!(s.step, aidaemon_client_protocol::SetupStep::Done))
    });

    on_key("mod+k", move |_| palette.update(|open| *open = !*open));
    // A new conversation from the keyboard keeps the project of the one on
    // screen: Ctrl+N in a project's work stays in that project.
    on_key("mod+n", move |_| {
        store.new_in_project(store.current_project())
    });
    on_key("mod+b", move |_| store.open_sidebar());
    on_key(steps_key(), move |_| store.open_inspector());
    // The changes of the conversation on screen.
    on_key("mod+d", move |_| store.show_changes());
    on_key("mod+g", move |_| store.take_suggestion());
    // Fires while typing too (an input would keep Esc to itself), and
    // still reaches the input after.
    on_key_with(
        "escape",
        KeyOptions {
            pass_through: true,
            ..always()
        },
        move |_| {
            // Esc closes what is open first (the terminal's palette does not
            // see it), then stops the turn.
            if palette.get_untracked() {
                palette.set(false);
                return;
            }
            if let Some(id) = store.selected()
                && store.is_running(&id)
            {
                store.stop(&id);
            }
        },
    );
    // Copy the last reply, anywhere.
    on_key_with("mod+shift+c", always(), move |_| copy_reply(store));
    if let Some(key) = SYSTEM_KEY {
        on_key(key, move |_| store.open_system());
    }
    if terminal {
        // One meaning per state: stop the turn; else press twice to quit.
        let armed = stored(0_u64);
        on_key_with("mod+c", always(), move |_| {
            if let Some(id) = store.selected_untracked()
                && store.is_running(&id)
            {
                store.stop(&id);
                return;
            }
            let now = crate::store::now_ms();
            if now.saturating_sub(armed.replace(now)) < 2_000 {
                if let Some(text) = store.transcript_markdown() {
                    crate::set_parting_words(text);
                }
                nsh_core::element::desktop::quit();
            } else {
                store.say("Press Ctrl+C again to quit");
            }
        });
        on_key_with("mod+y", always(), move |_| copy_reply(store));
    }

    // Only install each binding while that answer is offered for the open
    // conversation and the composer is empty. A focused input otherwise owns
    // these ordinary letters, rather than a global shortcut swallowing them.
    let approval_keys = if terminal {
        Some(
            view().children(aidaemon_client_core::client::APPROVAL_KEYS.into_iter().map(
                |(key, _)| {
                    when(
                        move || store.key_answer(key),
                        move || {
                            on_key_with(key, always(), move |_| {
                                store.answer_by_key(key);
                            });
                            view()
                        },
                    )
                },
            )),
        )
    } else {
        None
    };

    let main = view()
        .child(approval_keys)
        .flex_1()
        .flex_row()
        // Clear of the notch, the status bar and the home indicator (a
        // phone's browser in full screen, an installed web app), and of the
        // on-screen keyboard: what is typed stays in view above it. The
        // keyboard's height already covers the home indicator.
        .pt_safe_area()
        .px_safe_area()
        .pb(move || px(keyboard_height().0.max(safe_area().bottom.0)))
        // A narrow window shows one thing at a time: a pane, full-screen,
        // or the conversation.
        // A narrow window's panes are drawers over the page: each slides in
        // from its own edge over a dimmed page, and back out.
        .child(when(
            move || store.is_compact(),
            move || {
                view()
                    .child(ui::drawer(
                        Placement::Left,
                        move || store.sidebar_open(),
                        move || store.close_panes(),
                        sidebar(store),
                    ))
                    .child(ui::drawer(
                        Placement::Right,
                        move || store.details_open(),
                        move || store.close_panes(),
                        inspector(store),
                    ))
            },
        ))
        // A narrow window (a phone): the page alone, as wide as the screen.
        // A roomy one: the panes beside the page, each as wide as the owner
        // drags it. Only crossing between the two remakes the layout.
        .child(dynamic(move || {
            if store.is_compact() {
                return Some(untracked(|| {
                    view().flex_1().flex_row().child(page(store, in_setup))
                }));
            }
            Some(untracked(|| {
                view().flex_1().flex_row().child(
                    ui::split()
                        .flex_1()
                        .child(ui::pane(
                            ui::Edge::Left,
                            // A terminal's cells are wide: titles need more of them.
                            if terminal { 340.0 } else { ui::size::SIDEBAR },
                            200.0,
                            420.0,
                            move || store.sidebar_open() && roomy.get(),
                            sidebar(store),
                        ))
                        .child(ui::pane_rest(page(store, in_setup)))
                        .child(ui::pane(
                            ui::Edge::Right,
                            ui::size::INSPECTOR,
                            260.0,
                            520.0,
                            move || store.details_open() && wide.get(),
                            inspector(store),
                        )),
                )
            }))
        }))
        .child(when(palette, move || palette_dialog(store, palette)))
        .child(crate::features::sidebar::move_picker(store))
        .child(crate::ui::picture_viewer(
            store.viewing(),
            picture_actions(store),
        ))
        .child(crate::ui::live_screen(live_screen(store)))
        .child(ui::toasts(
            move || store.toast(),
            move || store.dismiss_toast(),
            move |session: &str| store.select(Some(session.to_owned())),
        ));
    if terminal {
        view().flex_1().child(main).child(key_line(store))
    } else {
        main
    }
}

/// The page beside the panes: setup, pairing, a page or a conversation.
fn page(store: Store, in_setup: Memo<bool>) -> impl IntoElement {
    // A row, as the page's own `flex_1` grows along it: in a column it would
    // take only the width its content asks for.
    view()
        .flex_1()
        .flex_row()
        .min_w(px(0.0))
        .child(dynamic(move || {
            if store.connection() == Connection::NoDaemon {
                return Some(untracked(|| {
                    crate::features::setup::where_page(store).key("where")
                }));
            }
            if in_setup.get() {
                return Some(untracked(|| {
                    crate::features::setup::page(store).key("setup")
                }));
            }
            if store.connection() == Connection::NeedsPairing {
                return Some(untracked(|| {
                    crate::features::pairing::pairing(store).key("pairing")
                }));
            }
            if store.page() == crate::store::Page::Memory {
                return Some(untracked(|| {
                    crate::features::memory::page(store).key("memory")
                }));
            }
            if store.page() == crate::store::Page::Settings {
                return Some(untracked(|| {
                    crate::features::settings::page(store).key("settings")
                }));
            }
            if store.page() == crate::store::Page::Automations {
                return Some(untracked(|| {
                    crate::features::automations::page(store).key("automations")
                }));
            }
            if store.page() == crate::store::Page::System {
                return Some(untracked(|| {
                    crate::features::system::page(store).key("system")
                }));
            }
            // A new page per conversation: its list and draft are its own.
            let selected = store.selected();
            Some(untracked(move || {
                conversation::page(store, selected.clone())
                    .key(format!("conversation-{}", selected.unwrap_or_default()))
            }))
        }))
}

/// A binding that fires while the composer has the keyboard too.
fn always() -> KeyOptions {
    KeyOptions {
        when: nsh_core::keymap::When::Always,
        ..KeyOptions::default()
    }
}

fn copy_reply(store: Store) {
    match store.last_reply() {
        Some(reply) => {
            set_clipboard(reply);
            store.say("Copied the last reply");
        }
        None => store.say("No reply to copy yet"),
    }
}

/// The terminal's last line: where the keys are, for the screen in front.
pub fn key_line(store: Store) -> View {
    let keys = move || {
        let mut keys = vec![
            (key_label("mod+k"), "commands"),
            (key_label("mod+n"), "new"),
            (key_label("mod+b"), "list"),
        ];
        // Only work on code has the details (changes, and steps beside).
        if store.coding_on_screen() {
            keys.push((key_label("mod+d"), "changes"));
            keys.push((key_label(steps_key()), "steps"));
        }
        if let Some(id) = store.selected()
            && store.is_running(&id)
        {
            keys.push((key_label("mod+c"), "stop"));
        } else {
            keys.push((key_label("mod+y"), "copy reply"));
            keys.push((format!("{} ×2", key_label("mod+c")), "quit"));
        }
        keys.into_iter()
            .map(|(key, what)| format!("{key} {what}"))
            .collect::<Vec<_>>()
            .join("  ·  ")
    };
    crate::ui::key_line(keys)
}

fn palette_dialog(store: Store, open: Signal<bool>) -> impl IntoElement {
    let mut items = vec![
        CommandItem::new("new", "New conversation")
            .group("Conversation")
            .shortcut("mod+n"),
        CommandItem::new("sidebar", "Show or hide the sidebar")
            .group("View")
            .shortcut("mod+b"),
        CommandItem::new("inspector", "Show or hide the changes and steps")
            .group("View")
            .shortcut(steps_key()),
        CommandItem::new("changes", "Show the changes of this conversation")
            .group("View")
            .shortcut("mod+d"),
    ];
    let system = CommandItem::new("system", "System: how the daemon is doing").group("View");
    items.push(match SYSTEM_KEY {
        Some(key) => system.shortcut(key),
        None => system,
    });
    items.push(
        CommandItem::new(
            "automations",
            "Automations: what runs on its own (pause, resume, run now)",
        )
        .group("View"),
    );
    items.push(CommandItem::new("memory", "Memory: what the daemon remembers").group("View"));
    items.push(CommandItem::new("settings", "Settings: change how the daemon runs").group("View"));
    items.push(
        CommandItem::new(
            "activity",
            format!(
                "Tool activity in replies: {} (change)",
                store.activity_mode().label()
            ),
        )
        .group("View"),
    );
    items.push(CommandItem::new("new-project", "New project").group("Projects"));
    if let Some(project) = store.current_project().and_then(|id| store.project(&id)) {
        items.push(
            CommandItem::new(
                "edit-project",
                format!("Edit {}: name or folder", project.name),
            )
            .group("Projects"),
        );
    }
    // Every waiting approval, answerable from the keyboard.
    for approval in store.approvals() {
        let what = approval.summary.clone();
        items.push(
            CommandItem::new(
                format!("approve:{}", approval.id),
                format!("Allow once: {what}"),
            )
            .group("Waiting for you"),
        );
        items.push(
            CommandItem::new(format!("deny:{}", approval.id), format!("Deny: {what}"))
                .group("Waiting for you"),
        );
        items.push(
            CommandItem::new(format!("show:{}", approval.id), format!("Show: {what}"))
                .group("Waiting for you"),
        );
    }
    for id in store.session_ids().into_iter().take(50) {
        items.push(
            CommandItem::new(format!("go:{id}"), store.title(&id))
                .group("Go to")
                .keywords([store.channel(&id).unwrap_or_default()]),
        );
    }
    dialog()
        .on_cancel(move || open.set(false))
        .items_center()
        .pt(px(96.0))
        .child(
            view()
                .w(px(560.0))
                .max_w(percentage(0.92))
                .rounded(px(radius::SURFACE))
                .border_1()
                .border_color(colors::EDGE)
                .bg(colors::SURFACE)
                .shadow_lg()
                .p_2()
                .child(
                    command()
                        .items(items)
                        .placeholder("Search conversations and commands…")
                        .accessibility_label("Commands")
                        .auto_focus(true)
                        // Escape clears the search, then closes the palette.
                        .on_cancel(move || open.set(false))
                        .on_select(move |id: &SharedString| {
                            open.set(false);
                            // Typing goes back to the composer, except
                            // where the command opens a form to type in.
                            if !matches!(id.as_ref(), "new-project" | "edit-project") {
                                store.refocus_composer();
                            }
                            match id.as_ref() {
                                "new" => store.new_in_project(store.current_project()),
                                "sidebar" => store.open_sidebar(),
                                "inspector" => store.open_inspector(),
                                "changes" => store.show_changes(),
                                "new-project" => {
                                    store.reveal_sidebar();
                                    store.show_project_form(true);
                                }
                                "edit-project" => {
                                    if let Some(id) = store.current_project() {
                                        store.reveal_sidebar();
                                        store.edit_project(&id);
                                    }
                                }
                                "system" => store.open_system(),
                                "memory" => store.open_memory(),
                                "automations" => store.open_automations(),
                                "settings" => store.open_settings(),
                                "activity" => store.cycle_activity_mode(),
                                "copy-reply" => copy_reply(store),
                                "copy-all" => {
                                    if let Some(text) = store.transcript_markdown() {
                                        set_clipboard(text);
                                        store.say("Copied the conversation");
                                    }
                                }
                                "pin" => {
                                    if let Some(id) = store.selected_untracked() {
                                        store.toggle_pin(&id);
                                    }
                                }
                                other => {
                                    if let Some(session) = other.strip_prefix("go:") {
                                        store.select(Some(session.to_owned()));
                                    } else if let Some(id) = other.strip_prefix("approve:") {
                                        store.resolve(id, ApprovalDecision::ApproveOnce);
                                    } else if let Some(id) = other.strip_prefix("deny:") {
                                        store.resolve(id, ApprovalDecision::Deny);
                                    } else if let Some(id) = other.strip_prefix("show:") {
                                        store.show_approval(id);
                                    }
                                }
                            }
                        })
                        .empty(
                            text("Nothing matches.")
                                .p_3()
                                .text_size(px(type_scale::SM))
                                .text_color(colors::MUTED),
                        )
                        .h(px(360.0))
                        .item_class("rounded-md focus:bg-accent selected:bg-accent")
                        .render_item(|entry: &CommandEntry| {
                            view()
                                .flex_row()
                                .items_center()
                                .h(px(size::ROW))
                                .px_2()
                                .gap_2()
                                .child(
                                    text(entry.label.clone())
                                        .flex_1()
                                        .number_of_lines(1)
                                        .text_size(px(type_scale::BASE)),
                                )
                                .child(
                                    text(entry.shortcut_label.clone().unwrap_or_default())
                                        .text_size(px(type_scale::XS))
                                        .text_color(colors::FAINT),
                                )
                        }),
                ),
        )
}

/// The System page's key. A browser keeps ⌘, for its own settings, and a
/// shifted comma arrives as "<", so the web build has none: the palette and
/// the sidebar reach the page there.
#[cfg(not(target_arch = "wasm32"))]
const SYSTEM_KEY: Option<&str> = Some("mod+,");
#[cfg(target_arch = "wasm32")]
const SYSTEM_KEY: Option<&str> = None;

/// The details pane's key. A terminal sends Ctrl+I as Tab, so there it is
/// Ctrl+O.
fn steps_key() -> &'static str {
    if idiom() == nsh_core::platform::Idiom::Terminal {
        "mod+o"
    } else {
        "mod+i"
    }
}

/// The live view of the agent's browser, read and driven through the store.
fn live_screen(store: Store) -> crate::ui::LiveScreen {
    use std::rc::Rc;
    crate::ui::LiveScreen {
        open: Rc::new(move || store.live_open()),
        title: Rc::new(move || {
            store.live_surface().map_or_else(
                || "The agent's browser".to_owned(),
                |s| match s.site {
                    Some(site) if !s.title.is_empty() => format!("{} · {site}", s.title),
                    Some(site) => site,
                    None => s.title,
                },
            )
        }),
        status: Rc::new(move || {
            if let Some(why) = store.live_error() {
                return why;
            }
            if store.live_controlling() {
                "You're in control: the agent waits until you give it back".to_owned()
            } else if store.live_surface().is_some_and(|s| s.handoff_pending) {
                "The agent is waiting for you: take control, then press Done".to_owned()
            } else {
                "Watching: the agent keeps working".to_owned()
            }
        }),
        frame: store.live_frame(),
        controlling: Rc::new(move || store.live_controlling()),
        focus: Rc::new(move || store.live_focus()),
        can_control: Rc::new(move || store.live_surface().is_some_and(|s| s.interactive)),
        waiting: Rc::new(move || store.live_surface().is_some_and(|s| s.handoff_pending)),
        pointer: Rc::new(move |action, x, y| store.live_pointer(action, x, y)),
        point: Rc::new(move |x, y| store.live_point(x, y)),
        can_point: Rc::new(move || store.live_surface().is_some_and(|s| s.session_id.is_some())),
        wheel: Rc::new(move |dy| store.live_wheel(dy)),
        text: Rc::new(move |text| store.live_text(text)),
        key: Rc::new(move |code, key| store.live_key(code, key)),
        take: Rc::new(move || store.live_take()),
        give_back: Rc::new(move || store.live_give_back()),
        done: Rc::new(move || store.live_finish(true)),
        not_now: Rc::new(move || store.live_finish(false)),
        close: Rc::new(move || store.close_live()),
    }
}

/// What the picture viewer offers. The browser saves a copy, shares it
/// through the system's sheet (where there is one) and draws marks over it
/// for the message; a window opens it in the system's browser.
fn picture_actions(store: Store) -> crate::ui::ViewerActions {
    use crate::ui::{Glyph, Viewed, ViewerAction, ViewerActions};
    use std::rc::Rc;
    #[cfg(target_arch = "wasm32")]
    {
        use crate::link::web;
        ViewerActions {
            save: ViewerAction {
                label: "Download",
                glyph: Glyph::Download,
                run: Rc::new(|shown: &Viewed| web::download(&shown.url, &shown.name)),
            },
            share: web::can_share().then(|| ViewerAction {
                label: "Share",
                glyph: Glyph::Share,
                run: Rc::new(|shown: &Viewed| web::share_file(&shown.url, &shown.name)),
            }),
            can_mark_up: Rc::new(move || store.takes_files()),
            mark_up: Rc::new(move |shown: &Viewed, strokes| store.attach_marked(shown, strokes)),
            opened: Rc::new(|shown: &Viewed| web::prefetch_share(&shown.url)),
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = store;
        ViewerActions {
            save: ViewerAction {
                label: "Open in browser",
                glyph: Glyph::External,
                run: Rc::new(|shown: &Viewed| drop(nsh_core::prelude::open_url(shown.url.clone()))),
            },
            share: None,
            can_mark_up: Rc::new(|| false),
            mark_up: Rc::new(|_: &Viewed, _| {}),
            opened: Rc::new(|_: &Viewed| {}),
        }
    }
}
