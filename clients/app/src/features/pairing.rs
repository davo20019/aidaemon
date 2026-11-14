//! Pairing a device: the code from `/pair_device` (or `aidaemon pair`), a
//! name the owner will recognise, and a wait while they confirm in the chat
//! they asked from.

use nsh_core::prelude::*;

use crate::link;
use crate::store::Store;
use crate::ui::{self, Size, Tone, Variant, btn, colors, field, status, type_scale};

pub fn pairing(store: Store) -> View {
    let offered = link::offered_pairing();
    let offer = signal(offered.as_ref().map(|o| o.0.clone()).unwrap_or_default());
    let code = signal(offered.as_ref().map(|o| o.1.clone()).unwrap_or_default());
    let fingerprint = offered.and_then(|o| o.2);
    let name = signal(String::from("Browser"));
    let state = signal(None::<(String, Tone)>);
    let busy = signal(false);
    let from_link = offer.with_untracked(|o| !o.is_empty());
    let submit = move || {
        if busy.get_untracked() {
            return;
        }
        busy.set(true);
        state.set(Some((
            "Confirm the pairing in the chat you asked from…".into(),
            Tone::Accent,
        )));
        let fingerprint = fingerprint.clone();
        spawn(async move {
            let result = link::pair(
                store,
                offer.get_untracked().trim().to_owned(),
                code.get_untracked().trim().to_owned(),
                name.get_untracked().trim().to_owned(),
                fingerprint,
            )
            .await;
            // On success the screen is already gone: its signals with it.
            let _ = busy.try_set(false);
            if let Err(why) = result {
                let _ = state.try_set(Some((why, Tone::Danger)));
            }
        });
    };
    view().flex_1().items_center().justify_center().p_6().child(
        ui::surface()
            .w(px(420.0))
            .max_w(percentage(1.0))
            .p_6()
            .gap_4()
            .child(
                text("Pair this device")
                    .text_size(px(type_scale::TITLE))
                    .font_weight(FontWeight::SEMIBOLD),
            )
            .child(
                text("Ask your agent for a pairing code with /pair_device, then open its link here or enter the code. You confirm the pairing in that same chat.")
                    .text_size(px(type_scale::SM))
                    .line_height(px(type_scale::LH_SM))
                    .text_color(colors::MUTED),
            )
            .child(when(!from_link, move || labeled("Offer", field("Offer id from the link", offer))))
            .child(labeled("Code", field("Pairing code", code)))
            .child(labeled("Device name", field("How the daemon will list this device", name)))
            .child(
                view()
                    .flex_row()
                    .items_center()
                    .gap_3()
                    .child(
                        btn("Pair", Variant::Primary, Size::Md, submit)
                            .disabled(move || busy.get() || code.with(|c| c.trim().is_empty())),
                    )
                    .child(dynamic(move || {
                        let (message, tone) = state.get()?;
                        Some(untracked(move || status(message, tone).flex_1().min_w(px(0.0))))
                    })),
            ),
    )
}

fn labeled(label: &'static str, control: impl IntoElement) -> View {
    view()
        .gap_1p5()
        .child(
            text(label)
                .text_size(px(type_scale::XS))
                .font_weight(FontWeight::MEDIUM)
                .text_color(colors::MUTED),
        )
        .child(control)
}
