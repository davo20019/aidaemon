//! The window: sidebar, conversation, inspector, and the command palette
//! over them. The terminal gets the same panes; narrow windows fold the
//! inspector away first, then the sidebar.

use aidaemon_client_protocol::ApprovalDecision;
use nsh_base::{CommandEntry, CommandItem, command, dialog};
use nsh_core::platform::Idiom;
use nsh_core::prelude::*;

use crate::features::{conversation, inspector::inspector, sidebar::sidebar};
use crate::store::{Connection, Store};
use crate::ui::{colors, radius, size, type_scale};

pub fn shell(store: Store) -> View {
    let terminal = idiom() == Idiom::Terminal;
    let wide = computed(move || window().width.0 >= 1180.0);
    let roomy = computed(move || window().width.0 >= 760.0);
    store.set_inspector(!terminal);
    effect(move || store.set_compact(!roomy.get()));
    crate::features::notify::approvals(store);
    crate::features::notify::replies(store);
    set_interval(std::time::Duration::from_secs(1), move || {
        store.advance_tick()
    });
    let palette = signal(false);

    on_key("mod+k", move |_| palette.update(|open| *open = !*open));
    on_key("mod+n", move |_| store.select(None));
    on_key("mod+b", move |_| store.open_sidebar());
    on_key("mod+i", move |_| store.open_inspector());
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
    on_key("mod+,", move |_| store.open_system());
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
                nsh_core::element::desktop::quit();
            } else {
                store.say("Press Ctrl+C again to quit");
            }
        });
        on_key_with("mod+y", always(), move |_| copy_reply(store));
    }

    let main = view()
        .flex_1()
        .flex_row()
        .child(
            view()
                .style_when(
                    move || !(store.sidebar_open() && roomy.get()),
                    |s| s.hidden(),
                )
                .child(sidebar(store)),
        )
        // A narrow window shows one thing at a time: a pane, full-screen,
        // or the conversation.
        .child(
            view()
                .flex_1()
                .style_when(
                    move || !(store.is_compact() && store.sidebar_open()),
                    |s| s.hidden(),
                )
                .child(sidebar(store)),
        )
        .child(
            view()
                .flex_1()
                .style_when(
                    move || !(store.is_compact() && store.inspector_open()),
                    |s| s.hidden(),
                )
                .child(inspector(store)),
        )
        .child(dynamic(move || {
            if store.is_compact() && (store.sidebar_open() || store.inspector_open()) {
                return None;
            }
            if store.connection() == Connection::NeedsPairing {
                return Some(untracked(|| {
                    crate::features::pairing::pairing(store).key("pairing")
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
        .child(
            view()
                .style_when(
                    move || !(store.inspector_open() && wide.get() && !store.is_compact()),
                    |s| s.hidden(),
                )
                .child(inspector(store)),
        )
        .child(when(palette, move || palette_dialog(store, palette)));
    if terminal {
        view().flex_1().child(main).child(key_line(store))
    } else {
        main
    }
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
            (key_label("mod+i"), "inbox"),
        ];
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
        CommandItem::new("inspector", "Show or hide the inbox")
            .group("View")
            .shortcut("mod+i"),
    ];
    items.push(
        CommandItem::new("system", "System: how the daemon is doing")
            .group("View")
            .shortcut("mod+,"),
    );
    items.push(
        CommandItem::new("copy-reply", "Copy the last reply")
            .group("Conversation")
            .shortcut("mod+shift+c"),
    );
    if store.selected_untracked().is_some() {
        items.push(
            CommandItem::new("copy-all", "Copy this conversation as Markdown")
                .group("Conversation"),
        );
        items.push(CommandItem::new("pin", "Pin or unpin this conversation").group("Conversation"));
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
                .border_color(colors::BORDER)
                .bg(colors::SURFACE)
                .shadow_lg()
                .p_2()
                .child(
                    command()
                        .items(items)
                        .placeholder("Search conversations and commands…")
                        .accessibility_label("Commands")
                        .auto_focus(true)
                        .on_select(move |id: &SharedString| {
                            open.set(false);
                            match id.as_ref() {
                                "new" => store.select(None),
                                "sidebar" => store.open_sidebar(),
                                "inspector" => store.open_inspector(),
                                "system" => store.open_system(),
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
