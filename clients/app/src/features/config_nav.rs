//! Shared top navigation bar for the four configuration & daemon management
//! pages (`Settings`, `Automations`, `Memory`, `System`). Keeps the sidebar
//! focused on conversations and projects while giving one rock-steady bar to
//! switch between all four pages.

use nsh_core::prelude::*;

use crate::store::{Page, Store};
use crate::ui::{self, Glyph, Size, colors};

/// Top header bar shared across `Settings`, `Automations`, `Memory` and
/// `System`: sidebar toggle (when compact or closed) and the four page links
/// in a horizontally scrollable pill strip. Refresh lives on pull-to-refresh
/// and the keyboard/command palette.
pub fn header(store: Store) -> View {
    let action_size = if ui::touch() { Size::Sm } else { Size::Md };
    view()
        .min_h(px(ui::fit::bar()))
        .flex_row()
        .items_center()
        .gap(px(4.0))
        .px(px(if ui::touch() { 6.0 } else { 12.0 }))
        .border_b_1()
        .border_color(colors::LINE)
        .child(when(
            move || store.is_compact() || !store.sidebar_open(),
            move || {
                ui::icon_btn(Glyph::Sidebar, "Conversations", action_size, move || {
                    store.open_sidebar()
                })
            },
        ))
        .child(
            scroll()
                .horizontal()
                .scrollbar(false)
                .flex_1()
                .min_w(px(0.0))
                .child(
                    view()
                        .role(AccessibilityRole::Tablist)
                        .label("Pages")
                        .flex_row()
                        .items_center()
                        .gap(px(4.0))
                        .child(ui::page_nav_item(
                            Glyph::Settings,
                            "Settings",
                            || SharedString::from("Settings: change how the daemon runs"),
                            move || store.page() == Page::Settings,
                            || false,
                            move || store.open_settings(),
                        ))
                        .child(ui::page_nav_item(
                            Glyph::Clock,
                            "Automations",
                            move || match store.automation_news() {
                                0 => SharedString::from("Automations: what runs on its own"),
                                n => SharedString::from(format!(
                                    "Automations: what runs on its own, {n} new"
                                )),
                            },
                            move || store.page() == Page::Automations,
                            move || store.automation_news() > 0,
                            move || store.open_automations(),
                        ))
                        .child(ui::page_nav_item(
                            Glyph::Memory,
                            "Memory",
                            || SharedString::from("Memory: what the daemon remembers"),
                            move || store.page() == Page::Memory,
                            || false,
                            move || store.open_memory(),
                        ))
                        .child(ui::page_nav_item(
                            Glyph::System,
                            "System",
                            || SharedString::from("System: how the daemon is doing"),
                            move || store.page() == Page::System,
                            || false,
                            move || store.open_system(),
                        )),
                ),
        )
}
