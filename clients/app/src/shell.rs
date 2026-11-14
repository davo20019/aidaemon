//! The window: sidebar, conversation, inspector, and the command palette
//! over them. The terminal gets the same panes; narrow windows fold the
//! inspector away first, then the sidebar.

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
    let show_sidebar = signal(true);
    let show_inspector = signal(!terminal);
    let palette = signal(false);

    on_key("mod+k", move |_| palette.update(|open| *open = !*open));
    on_key("mod+n", move |_| store.select(None));
    on_key("mod+b", move |_| show_sidebar.update(|s| *s = !*s));
    on_key("mod+i", move |_| show_inspector.update(|s| *s = !*s));
    on_key("escape", move |_| {
        if let Some(id) = store.selected()
            && store.is_running(&id)
        {
            store.stop(&id);
        }
    });

    view()
        .flex_1()
        .flex_row()
        .child(
            view()
                .style_when(move || !(show_sidebar.get() && roomy.get()), |s| s.hidden())
                .child(sidebar(store)),
        )
        .child(dynamic(move || {
            if store.connection() == Connection::NeedsPairing {
                return Some(untracked(|| {
                    crate::features::pairing::pairing(store).key("pairing")
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
                    move || !(show_inspector.get() && wide.get()),
                    |s| s.hidden(),
                )
                .child(inspector(store)),
        )
        .child(when(palette, move || {
            palette_dialog(store, palette, show_sidebar, show_inspector)
        }))
}

fn palette_dialog(
    store: Store,
    open: Signal<bool>,
    show_sidebar: Signal<bool>,
    show_inspector: Signal<bool>,
) -> impl IntoElement {
    let mut items = vec![
        CommandItem::new("new", "New conversation")
            .group("Conversation")
            .shortcut("mod+n"),
        CommandItem::new("sidebar", "Show or hide the sidebar")
            .group("View")
            .shortcut("mod+b"),
        CommandItem::new("inspector", "Show or hide the activity panel")
            .group("View")
            .shortcut("mod+i"),
    ];
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
                                "sidebar" => show_sidebar.update(|s| *s = !*s),
                                "inspector" => show_inspector.update(|s| *s = !*s),
                                other => {
                                    if let Some(session) = other.strip_prefix("go:") {
                                        store.select(Some(session.to_owned()));
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
