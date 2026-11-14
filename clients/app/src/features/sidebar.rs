//! The sidebar: a new conversation, the conversations with what each is
//! doing, and the connection.

use nsh_core::prelude::*;

use crate::store::{Connection, Store, now_ms};
use crate::ui::{self, Glyph, Tone, colors, icon, row, section_label, size, status, type_scale};
use aidaemon_client_core::machine::DropKind;

pub fn sidebar(store: Store) -> View {
    let ids = computed(move || store.session_ids());
    view()
        .w(px(size::SIDEBAR))
        .flex_1()
        .bg(colors::SIDEBAR)
        .border_r_1()
        .border_color(colors::LINE)
        .child(
            view()
                .h(px(44.0))
                .flex_row()
                .items_center()
                .px_3()
                .gap_2()
                .child(
                    text("AIdaemon")
                        .flex_1()
                        .text_size(px(type_scale::BASE))
                        .font_weight(FontWeight::SEMIBOLD),
                )
                .child(ui::kbd(key_label("mod+k"))),
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
        .child(section_label("Conversations").px_4())
        .child(
            scroll()
                .flex_1()
                .child(view().px_2().pb_2().gap(px(1.0)).child(each(
                    ids,
                    |id: &String| id.clone(),
                    move |id, _| session_row(store, id.get_untracked()),
                ))),
        )
        .child(ui::separator())
        .child(connection(store))
}

fn session_row(store: Store, id: String) -> nsh_base::ButtonElement {
    let (a, b, c, d, e) = (id.clone(), id.clone(), id.clone(), id.clone(), id.clone());
    row(
        move || store.selected().as_deref() == Some(a.as_str()),
        move || store.select(Some(b.clone())),
    )
    .child(
        text(move || if store.is_running(&c) { "●" } else { "○" })
            .w(px(size::ICON))
            .text_center()
            .text_size(px(8.0))
            .text_color(move || {
                if store.is_running(&e) {
                    colors::ACCENT
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
                text(move || store.title(&d))
                    .number_of_lines(1)
                    .text_size(px(type_scale::BASE)),
            )
            .child(
                text(move || store.channel(&id).unwrap_or_default())
                    .number_of_lines(1)
                    .text_size(px(type_scale::XS))
                    .text_color(colors::FAINT),
            ),
    )
}

fn connection(store: Store) -> View {
    view().px_4().py_3().child(dynamic(move || {
        let state = store.connection();
        Some(untracked(move || match state {
            Connection::Online => status("Connected to the daemon", Tone::Success),
            Connection::Connecting => status("Connecting…", Tone::Accent),
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
