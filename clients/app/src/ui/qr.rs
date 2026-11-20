//! A QR code a phone's camera reads: dark modules on white with a quiet
//! zone in either theme (scanners need the contrast), as line art in a
//! window and in a browser, as half-block cells in a terminal.

use nsh_core::prelude::*;

use super::theme::radius;

/// `content` as a QR code `side` points wide; `None` when it is too long
/// for one.
pub fn qr_code(content: &str, side: f32) -> Option<View> {
    let code = qrcode::QrCode::new(content.as_bytes()).ok()?;
    if idiom() == nsh_core::platform::Idiom::Terminal {
        let cells = code
            .render::<qrcode::render::unicode::Dense1x2>()
            .dark_color(qrcode::render::unicode::Dense1x2::Light)
            .light_color(qrcode::render::unicode::Dense1x2::Dark)
            .quiet_zone(true)
            .build();
        return Some(view().child(text(cells)));
    }
    let markup = code
        .render::<qrcode::render::svg::Color<'_>>()
        .dark_color(qrcode::render::svg::Color("#000000"))
        .light_color(qrcode::render::svg::Color("#ffffff"))
        .quiet_zone(true)
        .build();
    Some(
        view().rounded(px(radius::CONTROL)).overflow_hidden().child(
            svg(nsh_core::element::ImageSrc::svg_markup(&markup))
                .w(px(side))
                .h(px(side)),
        ),
    )
}
