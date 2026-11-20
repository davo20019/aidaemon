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
    // A link that carries its code pairs at once: opening it is the ask,
    // and the owner still confirms in the chat it came from.
    if from_link && code.with_untracked(|c| !c.trim().is_empty()) {
        submit();
    }
    view().flex_1().items_center().justify_center().p_6().child(
        ui::surface()
            .w(px(420.0))
            .max_w(percentage(1.0))
            .p_6()
            .gap_4()
            .child(
                text("Pair this device")
                    .text_size(px(ui::fit::title()))
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

/// Where pairing a phone from this computer stands.
#[derive(Clone, PartialEq)]
enum Phone {
    Idle,
    Starting,
    /// The QR code is up; nothing has redeemed it yet.
    Showing(aidaemon_client_protocol::PairingOffer),
    /// A device redeemed it and waits for the owner.
    Asking(aidaemon_client_protocol::PairingOffer, String),
    Ended(String, Tone),
}

/// Pair a phone from the daemon's own computer: a QR code of the offer's
/// `aidaemon-pair:` link (the phone's camera opens it in the app, which
/// pairs at once), then the owner approves the phone here.
pub fn pair_phone(store: Store) -> View {
    use aidaemon_client_protocol::EnrollmentState;
    use link::owner_pairing;

    let phase = signal(Phone::Idle);
    let start = move || {
        if phase.with_untracked(|p| matches!(p, Phone::Starting)) {
            return;
        }
        phase.set(Phone::Starting);
        spawn(async move {
            let offer = match owner_pairing::offer().await {
                Ok(offer) => offer,
                Err(why) => {
                    let _ = phase.try_set(Phone::Ended(why, Tone::Danger));
                    return;
                }
            };
            let id = offer.offer_id.clone();
            if !phase.try_set(Phone::Showing(offer.clone())) {
                return;
            }
            // Watch the offer while this one is on screen.
            loop {
                timer(std::time::Duration::from_secs(1)).await;
                let current = phase.try_get().is_some_and(|p| match p {
                    Phone::Showing(o) | Phone::Asking(o, _) => o.offer_id == id,
                    _ => false,
                });
                if !current {
                    return;
                }
                let Ok(enrollment) = owner_pairing::status(id.clone()).await else {
                    continue;
                };
                let next = match enrollment.state {
                    EnrollmentState::Open => continue,
                    EnrollmentState::Pending => {
                        if phase.with_untracked(|p| matches!(p, Phone::Asking(..))) {
                            continue;
                        }
                        Phone::Asking(
                            offer.clone(),
                            enrollment.device_name.unwrap_or_else(|| "A device".into()),
                        )
                    }
                    EnrollmentState::Approved => {
                        store.refresh_devices();
                        Phone::Ended("Paired.".into(), Tone::Success)
                    }
                    EnrollmentState::Denied => Phone::Ended("Not paired.".into(), Tone::Neutral),
                    _ => Phone::Ended("The code expired. Show a new one.".into(), Tone::Neutral),
                };
                if !phase.try_set(next) {
                    return;
                }
            }
        });
    };
    let decide = move |approve: bool| {
        let Some(id) = phase.with_untracked(|p| match p {
            Phone::Asking(o, _) => Some(o.offer_id.clone()),
            _ => None,
        }) else {
            return;
        };
        spawn(async move {
            let next = match owner_pairing::decide(id, approve).await {
                Ok(e) if e.state == EnrollmentState::Approved => {
                    store.refresh_devices();
                    Phone::Ended("Paired.".into(), Tone::Success)
                }
                Ok(_) => Phone::Ended("Not paired.".into(), Tone::Neutral),
                Err(why) => Phone::Ended(why, Tone::Danger),
            };
            let _ = phase.try_set(next);
        });
    };

    view().gap_2().child(dynamic(move || {
        let now = phase.get();
        Some(untracked(move || match now {
            Phone::Idle | Phone::Starting => view().flex_row().child(
                btn("Pair a phone", Variant::Secondary, Size::Sm, start)
                    .disabled(move || phase.with(|p| matches!(p, Phone::Starting))),
            ),
            Phone::Showing(offer) => {
                let mut card = ui::inset().gap_3().items_start();
                if !owner_pairing::reachable_elsewhere(&offer) {
                    card = card.child(ui::banner(
                        "A phone cannot reach AIDaemon yet: it listens only on this computer. \
                         Turn on remote access (Tailscale or a Cloudflare tunnel, \
                         [exposure.client_gateway] in config.toml) first.",
                        Tone::Warning,
                    ));
                }
                if let Some(qr) = ui::qr_code(&offer.link, 200.0) {
                    card = card.child(qr);
                }
                card.child(hint(
                    "Scan this with your phone's camera and open it in AIDaemon. \
                     Or paste the link into the app.",
                ))
                .child(hint(format!(
                    "Code {} · valid 10 minutes · daemon {}",
                    offer.code,
                    short(&offer.fingerprint)
                )))
                .child(
                    view()
                        .flex_row()
                        .gap_2()
                        .child(btn("Copy link", Variant::Ghost, Size::Sm, {
                            let link = offer.link.clone();
                            move || set_clipboard(link.clone())
                        }))
                        .child(btn("Cancel", Variant::Ghost, Size::Sm, move || {
                            phase.set(Phone::Idle)
                        })),
                )
            }
            Phone::Asking(_, name) => ui::inset()
                .gap_3()
                .child(text(format!("Pair “{name}” with this AIDaemon?")))
                .child(hint(
                    "It will see your chats and can send messages and answer approvals.",
                ))
                .child(
                    view()
                        .flex_row()
                        .gap_2()
                        .child(btn("Pair", Variant::Primary, Size::Sm, move || {
                            decide(true)
                        }))
                        .child(btn("Deny", Variant::Ghost, Size::Sm, move || decide(false))),
                ),
            Phone::Ended(message, tone) => view()
                .flex_row()
                .items_center()
                .gap_3()
                .child(status(message, tone))
                .child(btn("Pair another", Variant::Ghost, Size::Sm, start)),
        }))
    }))
}

fn hint(words: impl Into<SharedString>) -> Text {
    text(words.into())
        .text_size(px(type_scale::SM))
        .line_height(px(type_scale::LH_SM))
        .text_color(colors::MUTED)
}

/// The fingerprint's first characters, enough to tell daemons apart.
fn short(fingerprint: &str) -> &str {
    fingerprint.get(..8).unwrap_or(fingerprint)
}
